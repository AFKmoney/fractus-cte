"""train subpackage: lightweight training infrastructure (L8).

OnlineTrainer: chunk-wise online trainer for the ContinuousThoughtEngine.
    Supports gradient accumulation + SGD/AdamW/RMSprop. Head-partial training
    (loss on last position only) for speed.
"""

from .online import OnlineTrainer

__all__ = ["OnlineTrainer"]
