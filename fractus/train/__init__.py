"""train subpackage: lightweight training infrastructure (L8).

StateCarryTrainer: chunk-wise training carrying the linear-attention state (S,z)
    across chunk boundaries. O(chunk_len) memory instead of O(seq_len). The
    Mamba/RWKV trick, legitimate for linear attention.

LightweightTrainer: the same + AMP (bf16) + cosine LR scheduler + fused AdamW.

SurpriseGatedTrainer: energy-proportional training — backpropagate only on
    high-loss ('surprising') tokens. Skips tokens the model already knows.
"""

from .trainer import StateCarryTrainer, LightweightTrainer
from .surprise_gate import SurpriseGatedTrainer

__all__ = ["StateCarryTrainer", "LightweightTrainer", "SurpriseGatedTrainer"]
