"""nn subpackage: neural network modules (PyTorch).

Modules present in fractus-cte:
  - attention:    FractalLinearAttention (causal, linear, with S,z state carry)
  - phase_ode:    KuramotoLayer (coupled oscillator "consciousness clock")
  - moe:          PhaseRoutedMoE (von Mises gate, low-rank experts, self-modifying)
  - farey:        farey_sequence, expert_phases (phase distribution helpers)
  - stats:        elu_plus_one, stable_softmax
  - lazy_siren:   LazyStructuredSirenLinear (LoRA-style low-rank, no grid memory)
"""

from .stats import elu_plus_one, stable_softmax
from .attention import FractalLinearAttention
from .farey import farey_sequence, expert_phases
from .phase_ode import KuramotoLayer
from .moe import PhaseRoutedMoE

__all__ = [
    "elu_plus_one",
    "stable_softmax",
    "FractalLinearAttention",
    "farey_sequence",
    "expert_phases",
    "KuramotoLayer",
    "PhaseRoutedMoE",
]
