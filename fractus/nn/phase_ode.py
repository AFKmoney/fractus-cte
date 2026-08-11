"""KuramotoLayer: low-rank coupled Kuramoto oscillators, STATELESS.

Ported from the original system (src/phase_ode.rs) in pure PyTorch.

Math (low-rank form K = U*Lambda*U^T, RK4 integration):
    d theta_i/dt = omega_i - damping*theta_i + sum_j K_ij sin(theta_j - theta_i)
    Standard RK4 (4 sub-steps), then wrap theta_i mod 2*pi after each step.

STATELESS: no persistent state between forwards. Initial phases are derived from
hidden states at each call. U, Lambda, omega are nn.Parameter (coupling is learned).
"""

import math
import torch
import torch.nn as nn


class KuramotoLayer(nn.Module):
    """Low-rank Kuramoto oscillator layer, STATELESS.

    Args:
        d_model       : input dimension (hidden).
        n_oscillators : number of oscillators N.
        rank          : rank r of the low-rank coupling K = UΛUT.
        n_steps       : number of RK4 steps per forward.
        dt            : RK4 step size.
        damping       : linear damping (the -damping·θ term).
    """

    def __init__(
        self,
        d_model: int,
        n_oscillators: int,
        rank: int,
        n_steps: int = 4,
        dt: float = 0.1,
        damping: float = 0.01,
    ):
        super().__init__()
        if n_oscillators < 1 or rank < 1 or rank > n_oscillators:
            raise ValueError("n_oscillators >= 1 and 1 <= rank <= n_oscillators")
        self.d_model = d_model
        self.N = n_oscillators
        self.rank = rank
        self.n_steps = n_steps
        self.dt = dt
        self.damping = damping
        self.TWO_PI = 2.0 * math.pi

        # Trainable parameters (init as in the original phase_ode.rs:38-57).
        self.omega = nn.Parameter(torch.empty(n_oscillators).uniform_(-0.05, 0.05))
        self.coupling_u = nn.Parameter(torch.empty(n_oscillators, rank).uniform_(-1.0, 1.0))
        self.coupling_lambda = nn.Parameter(torch.empty(rank).uniform_(0.01, 0.51))

    def _derivative(self, theta: torch.Tensor) -> torch.Tensor:
        """dθ/dt for phases theta of shape (..., N). Low-rank form O(N·r)."""
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        p = torch.einsum("...n,nr->...r", sin_t, self.coupling_u)
        q = torch.einsum("...n,nr->...r", cos_t, self.coupling_u)
        u_p = torch.einsum("...r,nr->...n", self.coupling_lambda * p, self.coupling_u)
        u_q = torch.einsum("...r,nr->...n", self.coupling_lambda * q, self.coupling_u)
        dtheta = (
            self.omega
            - self.damping * theta
            + cos_t * u_p
            - sin_t * u_q
        )
        return dtheta

    def _rk4_step(self, theta: torch.Tensor, dt: float) -> torch.Tensor:
        """One RK4 step (4 derivative evals) unrolled inline — no Python loop,
        single fused autograd graph. The 4 sub-steps (k1→k2→k3→k4) are
        inherently sequential (each depends on the previous), so they stay
        unrolled-but-sequential. The win is killing interpreter round-trips."""
        k1 = self._derivative(theta)
        k2 = self._derivative(theta + 0.5 * dt * k1)
        k3 = self._derivative(theta + 0.5 * dt * k2)
        k4 = self._derivative(theta + dt * k3)
        return theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _rk4_integrate_looped(self, theta: torch.Tensor) -> torch.Tensor:
        """Reference implementation: looped, with mod-wrap after each step.
        Kept for the equivalence test (test_rk4_vectorized_matches_reference)."""
        dt = self.dt
        for _ in range(self.n_steps):
            theta = self._rk4_step(theta, dt)
            theta = torch.remainder(theta, self.TWO_PI)
        return theta

    def _rk4_integrate(self, theta: torch.Tensor) -> torch.Tensor:
        """Integrates n_steps RK4 steps from theta (..., N). Wraps mod 2π after
        each step (exact equivalence with the reference).

        L8 OPTIMIZATION: the n_steps outer loop is UNROLLED for the common
        n_steps<=4 case. This kills the Python `for` interpreter overhead
        between steps — the 4 sub-step graph (`_rk4_step`) is already inline.
        The mod-wrap is cheap (elementwise) and kept per-step so the dynamics
        match the reference bit-for-bit.
        """
        dt = self.dt
        two_pi = self.TWO_PI
        theta = torch.remainder(self._rk4_step(theta, dt), two_pi)
        if self.n_steps > 1:
            theta = torch.remainder(self._rk4_step(theta, dt), two_pi)
        if self.n_steps > 2:
            theta = torch.remainder(self._rk4_step(theta, dt), two_pi)
        if self.n_steps > 3:
            for _ in range(self.n_steps - 3):
                theta = torch.remainder(self._rk4_step(theta, dt), two_pi)
        return theta

    def _encode_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Initial phases from hidden states (B, L, d_model) → (B, L, N)."""
        hidden_mean = hidden.mean(dim=-1) * self.TWO_PI  # (B, L)
        offsets = torch.arange(self.N, dtype=hidden.dtype, device=hidden.device)
        offsets = offsets / self.N * self.TWO_PI
        theta_init = hidden_mean.unsqueeze(-1) + offsets.view(1, 1, self.N)
        return torch.remainder(theta_init, self.TWO_PI)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, L, d_model) → phases (B, L, N) after RK4."""
        theta = self._encode_from_hidden(hidden)
        return self._rk4_integrate(theta)

    def phase_loss(self, phases: torch.Tensor) -> torch.Tensor:
        """L = -(1/N2)·[cosθTK·cosθ + sinθTK·sinθ] (low-rank), averaged per token.

        Normalization: we divide by (B·L·N/N) = B·L (a scalar per token,
        not the strict 1/N2 formula of the original, which assumed a single token). This is
        consistent with batched usage.
        """
        cos_t = torch.cos(phases)
        sin_t = torch.sin(phases)
        uc = torch.einsum("bln,nr->blr", cos_t, self.coupling_u)
        us = torch.einsum("bln,nr->blr", sin_t, self.coupling_u)
        term_cos = (uc ** 2 * self.coupling_lambda).sum()
        term_sin = (us ** 2 * self.coupling_lambda).sum()
        N = self.N
        scale = phases.numel() / (N * N + 1e-12)
        return -(term_cos + term_sin) / scale

    def decode_to_bias(self, phases: torch.Tensor, d_model: int) -> torch.Tensor:
        """Sinusoidal positional encoding from the phases. (B,L,N) → (B,L,d_model).

        Not wired into FractalBlockFull.forward (L2b) — a utility method
        exposed for future use (e.g. injecting a Kuramoto positional bias
        into a given layer). Tested separately.
        """
        B, L, N = phases.shape
        idx = torch.arange(d_model, device=phases.device) % N
        phases_used = phases[..., idx]
        j = torch.arange(d_model, dtype=phases.dtype, device=phases.device)
        freq = (j // 2 + 1).view(1, 1, d_model)
        sin_part = torch.sin(freq * phases_used) / torch.sqrt(freq)
        cos_part = torch.cos(freq * phases_used) / torch.sqrt(freq)
        bias = torch.empty(B, L, d_model, dtype=phases.dtype, device=phases.device)
        bias[..., 0::2] = sin_part[..., 0::2]
        bias[..., 1::2] = cos_part[..., 1::2]
        return bias
