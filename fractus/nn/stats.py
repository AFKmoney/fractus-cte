"""Numerical utilities for fractus.

Ported from the original system (src/math/stats.rs) in pure PyTorch, differentiable.

elu_plus_one : strictly positive feature map for linear attention.
    φ(x, α) = x + 1              if x > 0
            = α(e^x - 1) + 1     otherwise
    With α=1 (default), φ is strictly positive (min e^x > 0 for x→-∞,
    = 1 at x=0). This positivity guarantees that the denominator of causal
    linear attention stays well-defined.

stable_softmax : softmax with max subtraction (no overflow).
"""

import torch


def elu_plus_one(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """ELU+1 strictly positive feature map, differentiable.

    Args:
        x : tensor of arbitrary shape.
        alpha : ELU coefficient (1.0 by default, as in the original).
    Returns:
        tensor of the same shape, strictly positive.
    """
    # We use the direct formula (differentiable via torch.where):
    # positive branch: x + 1; negative branch: alpha * (exp(x) - 1) + 1.
    pos = x + 1.0
    neg = alpha * (torch.exp(x) - 1.0) + 1.0
    return torch.where(x > 0, pos, neg)


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax (max subtraction).

    If the exponential sum is < 1e-10, returns the uniform 1/N
    (limit behavior inherited from the original stats.rs:56-57).
    """
    max_logits, _ = logits.max(dim=dim, keepdim=True)
    exp = torch.exp(logits - max_logits)
    denom = exp.sum(dim=dim, keepdim=True)
    # Limit behavior: uniform if denom ~ 0.
    uniform = torch.full_like(exp, 1.0 / exp.shape[dim])
    return torch.where(denom > 1e-10, exp / denom, uniform)
