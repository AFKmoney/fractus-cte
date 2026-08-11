"""Farey sequence and phase selection for phase-routed MoE.

Ported from the original system (src/math/farey.rs).

The Farey sequence F_n is the ordered set of irreducible fractions p/q in
[0, 1] with q <= n. It is generated iteratively by the mediant property.

For the MoE: we take F_{2E} (order twice the number of experts) and select
E angles uniformly among the fractions, converted to angles 2π·p/q ∈ [0, 2π).
This yields a dense, non-collapsing, deterministic phase distribution —
useful for von Mises routing.
"""

import math
from typing import List, Tuple


def farey_sequence(n: int) -> List[Tuple[int, int]]:
    """Generates the Farey sequence F_n as a list of (p, q) in ascending order.

    Algorithm via the mediant (as in farey.rs:18-49).
    F_n contains exactly 1 + Σ_{q=1}^{n} φ(q) terms (φ = Euler's totient).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    fractions: List[Tuple[int, int]] = []
    a, b = 0, 1
    c, d = 1, n
    fractions.append((a, b))
    while c <= n:
        k = (n + b) // d
        next_c = k * c - a
        next_d = k * d - b
        a, b = c, d
        c, d = next_c, next_d
        fractions.append((a, b))
    return fractions


def expert_phases(n_experts: int) -> List[float]:
    """Selects n_experts angles ∈ [0, 2π) from F_{2·n_experts}.

    As in farey.rs:53-64: we build F_{2E} (double order), then select
    E angles uniformly from the n_frac = len(F_{2E}) available fractions.
    """
    if n_experts < 1:
        raise ValueError("n_experts must be >= 1")
    fractions = farey_sequence(2 * n_experts)
    n_frac = len(fractions)
    angles_all = [2.0 * math.pi * p / q for (p, q) in fractions]
    phases: List[float] = []
    for i in range(n_experts):
        idx = min(int(i * n_frac / n_experts), n_frac - 1)
        phases.append(angles_all[idx])
    return phases
