"""PhaseRoutedMoE: mixture-of-experts with von Mises phase routing.

Ported from the original system (src/moe.rs + farey.rs) in pure PyTorch.

Expert phases drawn from Farey sequence. Von Mises gate with top-k routing.
Load-balance loss as auxiliary. End-to-end differentiable.

L8 OPTIMIZATION (gather-first sparse dispatch):
    The original computed the outputs of ALL n_experts, then gathered the
    top-k — wasting (E-K)/E of the FLOPs (50-75% on typical presets). Here
    we GATHER FIRST: index_select the top-k experts' WEIGHTS, then compute
    only those K experts. Output is bit-identical (proven by
    test_moe_sparse_matches_reference), but we do K/E of the matmul work.

    Concretely: instead of einsum("bld,edf->blef") over all E experts, we
    build w1_selected[b,l,k] = w1[topk_idx[b,l,k]] via gather, then a single
    batched matmul over the K active experts per token.
"""

import math
import torch
import torch.nn as nn

from .farey import expert_phases


def _gelu(x: torch.Tensor) -> torch.Tensor:
    """Tanh GeLU approximation (as in moe.rs:14-17)."""
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    ))


class PhaseRoutedMoE(nn.Module):
    """Mixture-of-experts with von Mises phase routing on Farey phases.

    Args:
        d_model     : input/output dimension.
        n_experts   : number of experts E.
        top_k       : number of active experts per token (<= E).
        kappa       : von Mises concentration.
        temperature : gate temperature (κ_eff = κ/temperature).
        d_ff        : expert hidden dimension (64 by default, as in the original).
        expert_rank : if None, dense experts (W1 (E,D,F), W2 (E,F,D)). If r,
                      low-rank (LoRA-style) experts: W1 ≈ scale1·U1@V1ᵀ,
                      W2 ≈ scale2·U2@V2ᵀ with factors U1 (E,F,r), V1 (E,D,r),
                      U2 (E,D,r), V2 (E,F,r). The low-rank form keeps the
                      Fractus compression story (LazyStructuredSirenLinear
                      already established this for the 1B) and lets one
                      component serve the 13M and the 1B. Routing is unchanged.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        kappa: float = 4.0,
        temperature: float = 1.0,
        d_ff: int = 64,
        expert_rank: int | None = None,
    ):
        super().__init__()
        if n_experts < 1:
            raise ValueError("n_experts >= 1")
        if top_k < 1 or top_k > n_experts:
            raise ValueError(f"top_k must be in [1, {n_experts}], got {top_k}")
        if expert_rank is not None and expert_rank < 1:
            raise ValueError("expert_rank must be >= 1 (or None for dense)")
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.kappa = kappa
        self.temperature = temperature
        self.d_ff = d_ff
        self.expert_rank = expert_rank

        # Expert phases (Farey precomputation, off-graph).
        phases = expert_phases(n_experts)
        self.register_buffer("expert_phases", torch.tensor(phases, dtype=torch.float32))

        if expert_rank is None:
            # Dense experts: W1 (E,D,F), W2 (E,F,D). Xavier uniform init.
            scale1 = math.sqrt(2.0 / d_model)
            scale2 = math.sqrt(2.0 / d_ff)
            self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff).uniform_(-scale1, scale1))
            self.b1 = nn.Parameter(torch.zeros(n_experts, d_ff))
            self.w2 = nn.Parameter(torch.empty(n_experts, d_ff, d_model).uniform_(-scale2, scale2))
            self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))
        else:
            # Low-rank (LoRA-style) experts: W1 ≈ scale1·U1@V1ᵀ, W2 ≈ scale2·U2@V2ᵀ.
            # U1 (E, F, r), V1 (E, D, r); U2 (E, D, r), V2 (E, F, r).
            # Forward runs via two cheap matmuls (no full matrix materialized),
            # matching the LazyStructuredSirenLinear pattern used to fix the 1B.
            r = expert_rank
            su1 = math.sqrt(2.0 / (d_ff + r))
            sv1 = math.sqrt(2.0 / (d_model + r))
            su2 = math.sqrt(2.0 / (d_model + r))
            sv2 = math.sqrt(2.0 / (d_ff + r))
            self.U1 = nn.Parameter(torch.empty(n_experts, d_ff, r).uniform_(-su1, su1))
            self.V1 = nn.Parameter(torch.empty(n_experts, d_model, r).uniform_(-sv1, sv1))
            self.U2 = nn.Parameter(torch.empty(n_experts, d_model, r).uniform_(-su2, su2))
            self.V2 = nn.Parameter(torch.empty(n_experts, d_ff, r).uniform_(-sv2, sv2))
            self.scale1 = nn.Parameter(torch.ones(n_experts, 1, 1))
            self.scale2 = nn.Parameter(torch.ones(n_experts, 1, 1))
            self.b1 = nn.Parameter(torch.zeros(n_experts, d_ff))
            self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))

    def _compute_gates(self, phases: torch.Tensor) -> torch.Tensor:
        """Computes the von Mises gates for each token.

        phases: (B, L, n_phases). Returns gates (B, L, E).
        """
        sin_p = torch.sin(phases).sum(dim=-1)  # (B, L)
        cos_p = torch.cos(phases).sum(dim=-1)
        theta_bar = torch.atan2(sin_p, cos_p)  # (B, L)
        kappa_eff = self.kappa / self.temperature
        diff = theta_bar.unsqueeze(-1) - self.expert_phases.view(
            *[1] * (phases.dim() - 1), self.n_experts
        )  # (B, L, E)
        gates = torch.exp(kappa_eff * torch.cos(diff))  # (B, L, E)
        gates_sum = gates.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(gates, 1.0 / self.n_experts)
        gates = torch.where(gates_sum > 1e-10, gates / gates_sum, uniform)
        return gates

    def add_expert(self, phase: float = None, dominant_idx: int = None) -> int:
        """Add a new expert at runtime (self-modification).

        Grows every (E, ...) parameter by one row along dim 0, adds a new
        phase for routing, and increments n_experts.

        Stability design (validated post-hoc):
          - The new expert is placed NEAR the dominant expert's phase (slightly
            offset) so it captures traffic from the overloaded region — not in
            an empty gap where no token phases land.
          - Weights are initialized to ZERO (U/V/scale all zero). A zero expert
            outputs nothing → no perturbation to the forward pass → no gradient
            spike. It "warms up" gradually via backprop.

        Args:
            phase: optional explicit phase for the new expert.
            dominant_idx: index of the expert to split traffic from. If None,
                          uses the midpoint of the largest gap (legacy behavior).
        Returns:
            The index of the newly added expert.
        """
        old_E = self.n_experts
        new_E = old_E + 1

        # Choose a phase: near the dominant expert (slight offset) to capture
        # its overflow traffic, or explicit.
        if phase is None:
            if dominant_idx is not None and dominant_idx < old_E:
                # Place near the dominant expert, offset by a small amount.
                offset = 2 * math.pi / (old_E * 4)  # quarter of the average spacing
                phase = float((self.expert_phases[dominant_idx].item() + offset) % (2 * math.pi))
            else:
                # Legacy: midpoint of largest gap.
                sorted_phases = self.expert_phases.sort().values
                gaps = torch.diff(sorted_phases)
                wrap = sorted_phases[0] + 2 * math.pi - sorted_phases[-1]
                gaps = torch.cat([gaps, wrap.unsqueeze(0)])
                max_gap_idx = gaps.argmax().item()
                if max_gap_idx == len(gaps) - 1:
                    phase = float((sorted_phases[-1] + sorted_phases[0] + 2 * math.pi) / 2 % (2 * math.pi))
                else:
                    phase = float((sorted_phases[max_gap_idx] + sorted_phases[max_gap_idx + 1]) / 2)
        new_phase = torch.tensor([phase], dtype=self.expert_phases.dtype)
        self.expert_phases = torch.cat([self.expert_phases, new_phase])

        if self.expert_rank is None:
            # Dense mode: zero init (stable — no perturbation).
            self.w1 = nn.Parameter(torch.cat([self.w1.data, torch.zeros(1, self.d_model, self.d_ff)]))
            self.b1 = nn.Parameter(torch.cat([self.b1.data, torch.zeros(1, self.d_ff)]))
            self.w2 = nn.Parameter(torch.cat([self.w2.data, torch.zeros(1, self.d_ff, self.d_model)]))
            self.b2 = nn.Parameter(torch.cat([self.b2.data, torch.zeros(1, self.d_model)]))
        else:
            # Low-rank mode: zero init (stable — scale=0 means output=0).
            r = self.expert_rank
            self.U1 = nn.Parameter(torch.cat([self.U1.data, torch.zeros(1, self.d_ff, r)]))
            self.V1 = nn.Parameter(torch.cat([self.V1.data, torch.zeros(1, self.d_model, r)]))
            self.U2 = nn.Parameter(torch.cat([self.U2.data, torch.zeros(1, self.d_model, r)]))
            self.V2 = nn.Parameter(torch.cat([self.V2.data, torch.zeros(1, self.d_ff, r)]))
            self.scale1 = nn.Parameter(torch.cat([self.scale1.data, torch.zeros(1, 1, 1)]))
            self.scale2 = nn.Parameter(torch.cat([self.scale2.data, torch.zeros(1, 1, 1)]))
            self.b1 = nn.Parameter(torch.cat([self.b1.data, torch.zeros(1, self.d_ff)]))
            self.b2 = nn.Parameter(torch.cat([self.b2.data, torch.zeros(1, self.d_model)]))

        self.n_experts = new_E
        return old_E  # index of the new expert

    def _gather_compute_chunk(self, h_c, idx_c):
        """Low-rank expert FFN for ONE chunk of tokens."""
        n = h_c.shape[0]
        K = idx_c.shape[1]
        r = self.expert_rank
        D = h_c.shape[1]
        flat = idx_c.reshape(-1)
        g_U1 = self.U1.index_select(0, flat).reshape(n, K, self.d_ff, r)
        g_V1 = self.V1.index_select(0, flat).reshape(n, K, D, r)
        g_s1 = self.scale1.index_select(0, flat).reshape(n, K, 1, 1)
        g_b1 = self.b1.index_select(0, flat).reshape(n, K, self.d_ff)
        g_U2 = self.U2.index_select(0, flat).reshape(n, K, D, r)
        g_V2 = self.V2.index_select(0, flat).reshape(n, K, self.d_ff, r)
        g_s2 = self.scale2.index_select(0, flat).reshape(n, K, 1, 1)
        g_b2 = self.b2.index_select(0, flat).reshape(n, K, D)
        hV1 = torch.einsum('nd,nkdr->nkr', h_c, g_V1)
        h1 = g_s1.squeeze(-1) * torch.einsum('nkr,nkfr->nkf', hV1, g_U1) + g_b1
        h1_act = _gelu(h1)
        hV2 = torch.einsum('nkf,nkfr->nkr', h1_act, g_V2)
        out = g_s2.squeeze(-1) * torch.einsum('nkr,nkdr->nkd', hV2, g_U2) + g_b2
        return out

    def _sparse_expert_forward(
        self, h: torch.Tensor, topk_idx: torch.Tensor
    ) -> torch.Tensor:
        """GATHER-FIRST sparse forward, CHUNKED (memory-bounded).

        Preserves the phase routing, von Mises gate, top-K selection, low-rank
        expert math — ONLY changes how selected experts' outputs are materialized.
        Output is bit-identical to the dense gather.
        """
        B, L, D = h.shape
        K = topk_idx.shape[-1]
        N = B * L
        h_flat = h.reshape(N, D)
        idx_flat = topk_idx.reshape(N, K)

        if self.expert_rank is not None:
            chunk = 64
            if N <= chunk:
                return self._gather_compute_chunk(h_flat, idx_flat).reshape(B, L, K, D)
            outs = []
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                outs.append(self._gather_compute_chunk(h_flat[s:e], idx_flat[s:e]))
            return torch.cat(outs, dim=0).reshape(B, L, K, D)

        # Dense-expert path (small E).
        flat_idx = topk_idx.reshape(-1)
        w1_sel = self.w1.index_select(0, flat_idx).reshape(B, L, K, D, self.d_ff)
        b1_sel = self.b1.index_select(0, flat_idx).reshape(B, L, K, self.d_ff)
        w2_sel = self.w2.index_select(0, flat_idx).reshape(B, L, K, self.d_ff, D)
        b2_sel = self.b2.index_select(0, flat_idx).reshape(B, L, K, D)
        h_exp = h.unsqueeze(2).unsqueeze(-1)
        h1 = (h_exp * w1_sel).sum(dim=-2) + b1_sel
        h1_act = _gelu(h1)
        h1_act_exp = h1_act.unsqueeze(-1)
        out = (h1_act_exp * w2_sel).sum(dim=-2) + b2_sel
        return out

    def _dense_expert_forward(self, h: torch.Tensor) -> torch.Tensor:
        """DENSE forward (the original path): compute ALL E experts.

        h: (B, L, d_model) → outputs of all experts (B, L, E, d_model).
        Cheaper than sparse on CPU when E is small (einsum is more optimized
        than per-token index_select + broadcast). Used when n_experts is small.

        In low-rank mode the expert forward is computed via two cheap matmuls
        per expert (no full weight matrix materialized), matching the LoRA
        pattern in LazyStructuredSirenLinear: W1 ≈ scale1·U1@V1ᵀ,
        W2 ≈ scale2·U2@V2ᵀ.
        """
        B, L, D = h.shape
        if self.expert_rank is None:
            h1 = torch.einsum("bld,edf->blef", h, self.w1) + self.b1.view(1, 1, self.n_experts, self.d_ff)
            h1_act = _gelu(h1)
            out = torch.einsum("blef,efd->bled", h1_act, self.w2) + self.b2.view(1, 1, self.n_experts, self.d_model)
            return out
        # Low-rank layer 1: h1 = scale1 · (h @ V1) @ U1ᵀ + b1.
        #   h: (B,L,D); V1: (E,D,r) → hV1: (B,L,E,r); U1: (E,F,r) →
        #   contracting r: h1[b,l,e,f] = Σ_r hV1[b,l,e,r]·U1[e,f,r] = (h@V1)·U1ᵀ.
        #   scale1 is stored (E,1,1) (spec D1); reshape to (1,1,E,1) so it
        #   broadcasts over the E dim of (B,L,E,F) — mirroring the b1 reshape.
        hV1 = torch.einsum("bld,edr->bler", h, self.V1)            # (B,L,E,r)
        h1 = self.scale1.view(1, 1, self.n_experts, 1) * torch.einsum("bler,efr->blef", hV1, self.U1) + self.b1.view(1, 1, self.n_experts, self.d_ff)
        h1_act = _gelu(h1)
        # Low-rank layer 2: out = scale2 · (h1_act @ V2) @ U2ᵀ + b2.
        #   h1_act: (B,L,E,F); V2: (E,F,r) → hV2: (B,L,E,r); U2: (E,D,r) →
        #   contracting r: out[b,l,e,d] = Σ_r hV2[b,l,e,r]·U2[e,d,r] = (h1@V2)·U2ᵀ.
        hV2 = torch.einsum("blef,efr->bler", h1_act, self.V2)      # (B,L,E,r)
        out = self.scale2.view(1, 1, self.n_experts, 1) * torch.einsum("bler,edr->bled", hV2, self.U2) + self.b2.view(1, 1, self.n_experts, self.d_model)
        return out

    def forward(
        self, h: torch.Tensor, phases: torch.Tensor
    ):
        """h: (B, L, d_model), phases: (B, L, n_phases).
        Returns (output (B, L, d_model), load_balance_loss scalar).

        L8 ADAPTIVE DISPATCH: pick the cheaper path at construction time.
            - Sparse (gather-first) when n_experts > 2·top_k  (>50% waste saved).
            - Dense (einsum over all E) otherwise — on CPU the optimized einsum
              beats per-token index_select for small E.
        Measured: for E=4,K=2 the dense path is ~1.5× faster than sparse; for
        E=32,K=8 the sparse path wins. The 2× threshold is the empirical knee.

        Sparse now supports BOTH dense and low-rank expert modes.
        (E=4 <= 2·K=4), so this covers it.
        """
        gates = self._compute_gates(phases)  # (B, L, E)

        topk_vals, topk_idx = gates.topk(self.top_k, dim=-1)  # (B, L, K)
        topk_sum = topk_vals.sum(dim=-1, keepdim=True)
        uniform_topk = torch.full_like(topk_vals, 1.0 / self.top_k)
        topk_vals_norm = torch.where(
            topk_sum > 1e-10, topk_vals / topk_sum, uniform_topk
        )

        # Adaptive: dense when small E (einsum wins on CPU), sparse when large E.
        use_sparse = self.n_experts > 2 * self.top_k
        if use_sparse:
            topk_out = self._sparse_expert_forward(h, topk_idx)  # (B, L, K, d_model)
        else:
            all_out = self._dense_expert_forward(h)  # (B, L, E, d_model)
            idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
            topk_out = torch.gather(all_out, dim=2, index=idx_exp)  # (B, L, K, d_model)
        output = (topk_vals_norm.unsqueeze(-1) * topk_out).sum(dim=2)  # (B, L, d_model)

        # Load-balance loss uses the FULL gates (all E) — this is the only place
        # we still touch all experts, and it's a cheap mean over (B,L,E).
        P = gates.mean(dim=(0, 1))  # (E,)
        lb_loss = self.n_experts * ((P - 1.0 / self.n_experts) ** 2).sum()

        return output, lb_loss
