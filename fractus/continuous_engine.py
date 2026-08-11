"""ContinuousThoughtEngine: a model that thinks in real-time, not token-by-token.

Multi-block architecture: the engine stacks N CTEBlocks, each with its own
attention state (S,z), Kuramoto phases, and PhaseRoutedMoE. The thought state
flows through the stack as a residual stream — each block refines the thought.

    h → [Block 0: attn → kuramoto → moe] → [Block 1: attn → kuramoto → moe] → ... → output

The attention state (S,z) is PER-BLOCK and carried across chunk boundaries
(continuous thought). The thought_state (residual stream) is shared.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn.attention import FractalLinearAttention
from .nn.phase_ode import KuramotoLayer
from .nn.stats import elu_plus_one
from .nn.moe import PhaseRoutedMoE


class CTEBlock(nn.Module):
    """One block of the Continuous Thought Engine.

    Owns: attention, kuramoto, moE, norms, and its persistent state buffers.
    The thought flows in as h, gets refined (attn → kuramoto → moE), flows out.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        n_levels: int,
        n_oscillators: int,
        coupling_rank: int,
        n_experts: int,
        top_k: int,
        expert_d_ff: int,
        siren_rank: int = 32,
    ):
        super().__init__()
        self.d_model = d_model

        # Attention.
        self.attn = FractalLinearAttention(d_model, n_heads, d_head, n_levels)
        self.norm_attn = nn.LayerNorm(d_model)

        # Kuramoto.
        self.kuramoto = KuramotoLayer(d_model, n_oscillators, coupling_rank,
                                      n_steps=1, dt=0.1)
        self.norm_kur = nn.LayerNorm(d_model)

        # MoE.
        self.n_experts = n_experts
        self.top_k = top_k
        self.expert_d_ff = expert_d_ff
        self.moe = PhaseRoutedMoE(
            d_model=d_model, n_experts=n_experts, top_k=top_k,
            kappa=4.0, d_ff=expert_d_ff,
            expert_rank=(siren_rank if siren_rank else None),
        )
        self.norm_moe = nn.LayerNorm(d_model)

        # Per-block persistent state.
        nH_dH = n_heads * d_head
        self.register_buffer("attn_S", torch.zeros(1, nH_dH, nH_dH))
        self.register_buffer("attn_z", torch.zeros(1, nH_dH))
        self.register_buffer("kuramoto_phases", torch.zeros(1, 1, n_oscillators))

    def reset_state(self, batch_size: int = 1):
        """Zero this block's persistent state."""
        device = self.attn_S.device
        d = self.attn.n_heads * self.attn.d_head
        self.attn_S = torch.zeros(batch_size, d, d, device=device)
        self.attn_z = torch.zeros(batch_size, d, device=device)
        self.kuramoto_phases = torch.zeros(
            batch_size, 1, self.kuramoto.N, device=device)

    def tick_single(self, h: torch.Tensor) -> torch.Tensor:
        """Process a single-token thought state h: (B, 1, d_model) → (B, 1, d_model)."""
        B = h.shape[0]
        attn = self.attn
        D = attn.d_head

        # Attention: update (S, z) and read out.
        h_normed = self.norm_attn(h)
        q = torch.einsum("bld,de->ble", h_normed, attn.w_qkv[0]) + attn.b_qkv[0]
        k = torch.einsum("bld,de->ble", h_normed, attn.w_qkv[1]) + attn.b_qkv[1]
        v = torch.einsum("bld,de->ble", h_normed, attn.w_qkv[2]) + attn.b_qkv[2]
        q_feat = elu_plus_one(q + attn.level_offsets[0])
        k_feat = elu_plus_one(k + attn.level_offsets[0])

        for hd in range(attn.n_heads):
            kh = k_feat[:, :, hd * D:(hd + 1) * D]
            vh = v[:, :, hd * D:(hd + 1) * D]
            qh = q_feat[:, 0, hd * D:(hd + 1) * D]
            s_start = hd * D
            s_end = (hd + 1) * D
            outer = (kh.squeeze(1).unsqueeze(2) * vh.squeeze(1).unsqueeze(1))
            self.attn_S = self.attn_S.clone()
            self.attn_S[:, s_start:s_end, s_start:s_end] += outer.detach()
            self.attn_z = self.attn_z.clone()
            self.attn_z[:, s_start:s_end] += kh.squeeze(1).detach()

        attn_out = torch.zeros_like(h)
        for hd in range(attn.n_heads):
            s_start = hd * D
            s_end = (hd + 1) * D
            qh = q_feat[:, 0, hd * D:(hd + 1) * D]
            S_h = self.attn_S[:, s_start:s_end, s_start:s_end]
            z_h = self.attn_z[:, s_start:s_end]
            num = torch.bmm(qh.unsqueeze(1), S_h).squeeze(1)
            denom = (qh * z_h).sum(dim=-1, keepdim=True)
            safe = denom.abs() > 1e-10
            yh = torch.where(safe, num / (denom + 1e-20), torch.zeros_like(num))
            attn_out[:, 0, hd * D:(hd + 1) * D] = yh
        attn_out = attn_out @ attn.w_out + attn.b_out
        h = h + attn_out

        # Kuramoto: advance phases by one Euler step.
        h_kur = self.norm_kur(h)
        theta = self.kuramoto._encode_from_hidden(h_kur)
        theta = theta + 0.1 * self.kuramoto._derivative(theta)
        theta = torch.remainder(theta, self.kuramoto.TWO_PI)
        self.kuramoto_phases = theta.detach()

        # MoE: transform the thought, routed by Kuramoto phases.
        h_flat = h[:, 0, :]
        h_moe = self.norm_moe(h_flat).unsqueeze(1)
        phases_in = theta[:, 0:1, :]
        moe_out, lb_loss = self.moe(h_moe, phases_in)
        h = h + moe_out

        return h, lb_loss

    def tick_chunk_core(self, h: torch.Tensor) -> tuple:
        """Process a chunk thought state h: (B, C, d_model) → (B, C, d_model).

        Returns (h_transformed, lb_loss). Carries (S,z) across chunk boundaries.
        """
        B, C, D_model = h.shape
        attn = self.attn
        nH, dH, nL = attn.n_heads, attn.d_head, attn.n_levels

        # Attention: QKV projections + multi-level + carry.
        h_normed = self.norm_attn(h)
        q_all = torch.einsum("bld,de->ble", h_normed, attn.w_qkv[0]) + attn.b_qkv[0]
        k_all = torch.einsum("bld,de->ble", h_normed, attn.w_qkv[1]) + attn.b_qkv[1]
        v_all = torch.einsum("bld,de->ble", h_normed, attn.w_qkv[2]) + attn.b_qkv[2]
        q_all = q_all.view(B, C, nH, dH)
        k_all = k_all.view(B, C, nH, dH)
        v_all = v_all.view(B, C, nH, dH)

        offsets = attn.level_offsets
        q_lev = q_all.unsqueeze(1) + offsets.view(nL, 1, 1, 1)
        k_lev = k_all.unsqueeze(1) + offsets.view(nL, 1, 1, 1)
        q_feat = elu_plus_one(q_lev, alpha=1.0)
        k_feat = elu_plus_one(k_lev, alpha=1.0)
        v_lev = v_all.unsqueeze(1).expand(B, nL, C, nH, dH)
        q_flat = q_feat.permute(0, 1, 3, 2, 4).reshape(B * nL * nH, C, dH)
        k_flat = k_feat.permute(0, 1, 3, 2, 4).reshape(B * nL * nH, C, dH)
        v_flat = v_lev.permute(0, 1, 3, 2, 4).reshape(B * nL * nH, C, dH)

        # CARRY (S, z) — continuous thought across chunk boundaries.
        carry_S_per_head = torch.stack([
            self.attn_S[:, hd * dH:(hd + 1) * dH, hd * dH:(hd + 1) * dH]
            for hd in range(nH)
        ], dim=1)
        carry_z_per_head = torch.stack([
            self.attn_z[:, hd * dH:(hd + 1) * dH]
            for hd in range(nH)
        ], dim=1)
        carry_S_flat = carry_S_per_head.unsqueeze(1).expand(
            B, nL, nH, dH, dH).reshape(B * nL * nH, dH, dH)
        carry_z_flat = carry_z_per_head.unsqueeze(1).expand(
            B, nL, nH, dH).reshape(B * nL * nH, dH)

        y_flat, (S_final, z_final) = attn._linear_attention_causal_vectorized(
            q_flat, k_flat, v_flat, carry=(carry_S_flat, carry_z_flat))

        # Save final state: rebuild block-diagonal (B, nH*dH, nH*dH).
        S_reshaped = S_final.reshape(B, nL, nH, dH, dH).mean(dim=1)
        z_reshaped = z_final.reshape(B, nL, nH, dH).mean(dim=1)
        new_S = torch.zeros(B, nH * dH, nH * dH, device=h.device, dtype=h.dtype)
        new_z = torch.zeros(B, nH * dH, device=h.device, dtype=h.dtype)
        for hd in range(nH):
            new_S[:, hd * dH:(hd + 1) * dH, hd * dH:(hd + 1) * dH] = S_reshaped[:, hd]
            new_z[:, hd * dH:(hd + 1) * dH] = z_reshaped[:, hd]
        self.attn_S = new_S.detach()
        self.attn_z = new_z.detach()

        y = y_flat.reshape(B, nL, nH, C, dH).permute(0, 1, 3, 2, 4).reshape(B, nL, C, nH * dH)
        level_weights = torch.softmax(attn.level_logits, dim=-1)
        attn_out = (y * level_weights.view(1, nL, 1, 1)).sum(dim=1)
        attn_out = attn_out @ attn.w_out + attn.b_out
        h = h + attn_out

        # Kuramoto (detached — clock, not learned).
        with torch.no_grad():
            h_kur = self.norm_kur(h)
            theta = self.kuramoto._encode_from_hidden(h_kur)
            theta = self.kuramoto._rk4_integrate(theta)
        self.kuramoto_phases = theta

        # MoE.
        h_moe = self.norm_moe(h)
        phases_last = theta[:, -1:, :]
        phases_in = phases_last.expand(-1, C, -1)
        moe_out, lb_loss = self.moe(h_moe, phases_in)
        h = h + moe_out

        return h, lb_loss


class ContinuousThoughtEngine(nn.Module):
    """A continuous-time reasoning engine with multi-block depth.

    The thought state flows through N CTEBlocks (residual stream), each refining
    it with attention + Kuramoto + MoE. State (S,z) is per-block and continuous
    across chunk boundaries.

    Args:
        vocab_size:    vocabulary size.
        d_model:       dimension of the thought state.
        n_layers:      number of CTEBlocks (depth). Default 1 = retrocompatible.
        n_heads, d_head: attention configuration.
        n_levels:      attention levels.
        n_oscillators: Kuramoto oscillator count.
        coupling_rank: Kuramoto coupling rank.
        n_experts:     MoE expert count per block.
        top_k:         active experts per tick.
        expert_d_ff:   MoE expert hidden dim.
        siren_rank:    low-rank expert dimension (0 = dense).
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        n_layers: int = 1,
        n_heads: int = 4,
        d_head: int = 64,
        n_levels: int = 2,
        n_oscillators: int = 16,
        coupling_rank: int = 8,
        n_experts: int = 8,
        top_k: int = 2,
        expert_d_ff: int = 256,
        siren_rank: int = 32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        # Input embedding.
        self.observe = nn.Embedding(vocab_size, d_model)

        # Multi-block stack.
        self.blocks = nn.ModuleList([
            CTEBlock(
                d_model=d_model, n_heads=n_heads, d_head=d_head,
                n_levels=n_levels, n_oscillators=n_oscillators,
                coupling_rank=coupling_rank, n_experts=n_experts,
                top_k=top_k, expert_d_ff=expert_d_ff, siren_rank=siren_rank,
            )
            for _ in range(n_layers)
        ])

        # Load-balance loss accumulator.
        self.register_buffer("last_lb_loss", torch.tensor(0.0))

        # Heads.
        self.confidence_head = nn.Linear(d_model, 1)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.output_head.weight = self.observe.weight  # tied
        self.salience_head = nn.Linear(d_model, 1)

        # Memory.
        self.memory = None
        self.memory_active = True

        # Shared thought state (residual stream).
        self.register_buffer("thought_state", torch.zeros(1, 1, d_model))

    def reset_thought(self, batch_size: int = 1):
        """Reset the thought state and all per-block states to zero."""
        self.thought_state = torch.zeros(batch_size, 1, self.d_model,
                                         device=self.thought_state.device)
        for blk in self.blocks:
            blk.reset_state(batch_size)
        self._tick_count = getattr(self, '_tick_count', 0)
        self._expert_hits = getattr(self, '_expert_hits',
                                    torch.zeros(self.blocks[0].moe.n_experts))

    def attach_memory(self, memory):
        """Attach a PersistentMemory bank."""
        self.memory = memory

    def detach_memory(self):
        """Detach the memory bank."""
        self.memory = None

    def maybe_grow(self, *, min_ticks_between_grows: int = 1000,
                   max_experts: int = 32, imbalance_threshold: float = 0.8) -> bool:
        """Check if the model should grow a new expert (self-modification)."""
        if self.blocks[0].moe.n_experts >= max_experts:
            return False
        if self._tick_count - getattr(self, '_last_grow_tick', 0) < min_ticks_between_grows:
            return False
        if self._expert_hits.numel() != self.blocks[0].moe.n_experts:
            self._expert_hits = torch.zeros(self.blocks[0].moe.n_experts)
            return False
        total = self._expert_hits.sum().item()
        if total < self.blocks[0].moe.n_experts * 10:
            return False
        dominance = self._expert_hits.max().item() / max(total, 1)
        if dominance < imbalance_threshold:
            return False
        dominant_idx = self._expert_hits.argmax().item()
        for blk in self.blocks:
            blk.moe.add_expert(dominant_idx=dominant_idx)
        self._last_grow_tick = self._tick_count
        self._expert_hits = torch.zeros(self.blocks[0].moe.n_experts)
        print(f"[Fractus] Self-modified: grew expert in all {self.n_layers} blocks "
              f"(now {self.blocks[0].moe.n_experts} experts, dominance was {dominance:.2f})", flush=True)
        return True

    def tick(self, observation: torch.Tensor = None) -> tuple:
        """Advance the thought by ONE tick through all blocks.

        Returns: (output_logits, confidence).
        """
        B = self.thought_state.shape[0]
        h = self.thought_state  # (B, 1, d_model)

        if observation is not None:
            obs_vec = self.observe(observation).unsqueeze(1)
            h = h + obs_vec

        total_lb = torch.tensor(0.0, device=h.device)
        for blk in self.blocks:
            h, lb = blk.tick_single(h)
            total_lb = total_lb + lb.detach()
        self.last_lb_loss = total_lb

        # Track routing for self-modification.
        self._tick_count = getattr(self, '_tick_count', 0) + 1
        if hasattr(self, '_expert_hits') and self._expert_hits.numel() == self.blocks[0].moe.n_experts:
            with torch.no_grad():
                gates = self.blocks[0].moe._compute_gates(
                    self.blocks[0].kuramoto_phases[:, 0:1, :])
                topk_idx = gates.topk(self.blocks[0].moe.top_k, dim=-1).indices
                for e in range(self.blocks[0].moe.n_experts):
                    self._expert_hits[e] += (topk_idx == e).sum().item()

        # Update thought state.
        self.thought_state = h.detach().clone()

        # Memory: salience-gated consolidation + continuous injection.
        if self.memory is not None and self.memory_active:
            salience = torch.sigmoid(self.salience_head(h[:, 0, :]))
            self.memory.consolidate_if_salient(h[0:1, 0, :], salience[0].item())
            perturbation = self.memory.inject(self, blend=0.05, top_k=3)
            if not hasattr(self, '_pert_max'):
                self._pert_max = 1.0
            if not hasattr(self, '_prev_pert_target'):
                self._prev_pert_target = 0.0
            if perturbation > self._pert_max:
                self._pert_max = perturbation
            self.last_salience_loss = torch.nn.functional.binary_cross_entropy(
                salience[0:1, 0], torch.tensor([self._prev_pert_target]))
            self._prev_pert_target = min(perturbation / max(self._pert_max, 1e-8), 1.0)
        else:
            self.last_salience_loss = torch.tensor(0.0)

        # Confidence + output.
        confidence = torch.sigmoid(self.confidence_head(h[:, 0, :]).squeeze(-1))
        output_logits = self.output_head(h[:, 0, :])
        return output_logits, confidence

    def tick_chunk(self, observations: torch.Tensor) -> torch.Tensor:
        """Process a CHUNK of tokens through all blocks. Returns logits (B, C, vocab)."""
        B, C = observations.shape
        D = self.d_model

        obs_vecs = self.observe(observations)
        h = obs_vecs.clone()
        h[:, 0, :] = h[:, 0, :] + self.thought_state[:, 0, :]

        total_lb = torch.tensor(0.0, device=h.device)
        for blk in self.blocks:
            h, lb = blk.tick_chunk_core(h)
            total_lb = total_lb + lb.detach()
        self.last_lb_loss = total_lb

        self.thought_state = h[:, -1:, :].detach()
        output_logits = self.output_head(h)
        return output_logits

    def tick_chunk_train(self, observations: torch.Tensor) -> torch.Tensor:
        """Fast training: head on LAST position only. Returns logits (B, vocab)."""
        B, C = observations.shape

        obs_vecs = self.observe(observations)
        h = obs_vecs.clone()
        h[:, 0, :] = h[:, 0, :] + self.thought_state[:, 0, :]

        total_lb = torch.tensor(0.0, device=h.device)
        for blk in self.blocks:
            h, lb = blk.tick_chunk_core(h)
            total_lb = total_lb + lb.detach()
        self.last_lb_loss = total_lb

        self.thought_state = h[:, -1:, :].detach()
        last_logits = self.output_head(h[:, -1, :])
        return last_logits

    def think(self, observations: torch.Tensor, max_ticks: int = 10,
              confidence_threshold: float = 0.7) -> torch.Tensor:
        """Process observations with adaptive thinking depth."""
        B = observations.shape[0]
        outputs = []
        for t in range(observations.shape[1]):
            obs = observations[:, t]
            for tick in range(max_ticks):
                logits, conf = self.tick(obs if tick == 0 else None)
                if conf.mean().item() > confidence_threshold:
                    break
            outputs.append(logits)
        return torch.stack(outputs, dim=1)
