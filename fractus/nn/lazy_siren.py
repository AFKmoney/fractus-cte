"""LazyStructuredSirenLinear: the RAM-fix for the 1B model.

THE FIX. The original CachedStructuredSirenLinear stored a coordinate grid
of shape (out × in) per expert — for d_model=768, d_ff=1024 that's 786k
floats PER expert × 64 experts × 2 (w1+w2) × 8 layers = 800M floats = 3.2 GB
just for the grids. That's why the 1B OOMed.

This module NEVER materializes the full grid. Instead:

    1. The low-rank part U @ V^T is computed on-the-fly (cheap: (B,L,D) @ (D,r) @ (r,d_ff)).
       Cost: O(B·L·D·r) — tiny for rank=16.
    2. The SIREN residual is evaluated ONLY on the needed positions.
       Instead of pre-evaluating all (out × in) grid points, we evaluate
       the SIREN at a SMALL set of "anchor" coordinates and interpolate.

    Actually, simpler and more correct: we DROP the SIREN residual entirely
    for large matrices and use ONLY the low-rank decomposition W = U @ V^T.
    This is the LoRA approach — proven, fast, and uses O((D+d_ff)·r) params
    with ZERO grid memory.

    For the 1B model this means:
    - Each expert: 2 × (D + d_ff) × rank params (e.g. 2 × (768+1024) × 16 = 57k)
    - No grid, no SIREN evaluation, no reconstruction.
    - Forward: y = x @ (U @ V^T)^T = (x @ V) @ U^T — two cheap matmuls.
    - RAM: O(1) per expert (just the U,V parameters).

The "capacity" comes from having 64 experts × 8 layers = 512 low-rank
matrices, each representing a different subspace. The routing selects which
2 subspaces process each token.

This trades some expressiveness (no high-frequency residual) for MASSIVE
memory and speed gains. The right trade-off for CPU training of a large model.
"""

import math
import torch
import torch.nn as nn


class LazyStructuredSirenLinear(nn.Module):
    """Low-rank linear layer with NO grid memory (LoRA-style).

    W = scale * U @ V^T where U (out, r), V (in, r).
    Forward: y = (x @ V) @ U^T + b — two small matmuls, no full matrix.

    Args:
        in_features:  input dim.
        out_features: output dim.
        rank:         low-rank dimension.
        bias:         add trainable bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        bias: bool = True,
        **kwargs,  # absorb unused args (siren_hidden, refresh_every, etc.)
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self._call_count = 0
        self.refresh_every = 1  # no cache to refresh

        # Low-rank factors: W ≈ U @ V^T.
        scale_u = math.sqrt(2.0 / (out_features + rank))
        scale_v = math.sqrt(2.0 / (in_features + rank))
        self.U = nn.Parameter(torch.empty(out_features, rank).uniform_(-scale_u, scale_u))
        self.V = nn.Parameter(torch.empty(in_features, rank).uniform_(-scale_v, scale_v))

        # Scaling factor (learnable, like LoRA alpha).
        self.scale = nn.Parameter(torch.tensor(1.0))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Compatibility: fake _cached_W for code that references it.
        # But we DON'T store the full matrix — just return a small dummy.
        # The MoE code uses _cached_W for the stack — we override the MoE
        # to use U,V directly instead.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) → (..., out_features).

        y = scale * (x @ V) @ U^T + b
        Two matmuls of size (..., in, r) and (..., r, out).
        No full (in, out) matrix ever materialized.
        """
        h = x @ self.V  # (..., rank)
        y = self.scale * (h @ self.U.T)  # (..., out)
        if self.bias is not None:
            y = y + self.bias
        return y

    def reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct the full W (for inspection/export only)."""
        return self.scale * (self.U @ self.V.T)

    @property
    def _cached_W(self):
        """Compatibility property — returns reconstructed W (lazy, not stored)."""
        return self.reconstruct_weight()

    def force_refresh(self):
        """No-op (no cache to refresh)."""
        self._call_count = 0

    @property
    def n_dense_equivalent(self) -> int:
        return self.in_features * self.out_features

    @property
    def n_actual_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def compression_ratio(self) -> float:
        return self.n_dense_equivalent / max(self.n_actual_params, 1)
