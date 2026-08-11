"""nn subpackage: neural network modules (PyTorch).

L1: fractal embedding (FractalEmbedding).
L2a: causal linear attention (FractalLinearAttention) + minimal block (FractalBlock).
L2b: Kuramoto (KuramotoLayer) + von Mises/Farey MoE (PhaseRoutedMoE) + full block (FractalBlockFull).
"""

from .char_features import CharClassFeatures
from .fourier import MandelbrotFourierBasis
from .embedding import FractalEmbedding
from .stats import elu_plus_one, stable_softmax
from .attention import FractalLinearAttention
from .farey import farey_sequence, expert_phases
from .phase_ode import KuramotoLayer
from .moe import PhaseRoutedMoE
from .block import FractalBlock, FractalBlockFull
from .siren import TorusSirenWeight
from .siren_linear import SirenLinear

__all__ = [
    "CharClassFeatures",
    "MandelbrotFourierBasis",
    "FractalEmbedding",
    "elu_plus_one",
    "stable_softmax",
    "FractalLinearAttention",
    "farey_sequence",
    "expert_phases",
    "KuramotoLayer",
    "PhaseRoutedMoE",
    "FractalBlock",
    "FractalBlockFull",
    "TorusSirenWeight",
    "SirenLinear",
]
