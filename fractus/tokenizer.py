"""BPE tokenizer for Fractus-1B (byte-level, GPT-2 compatible).

Wraps the HuggingFace `tokenizers` library. Builds or loads a byte-level BPE
tokenizer with vocab_size ~50k, suitable for multi-language code/math/text.

Usage:
    tok = FractusTokenizer.build_or_load()
    ids = tok.encode("def hello(): pass")
    text = tok.decode(ids)
"""

import os
from typing import List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing


DEFAULT_VOCAB_SIZE = 50257
DEFAULT_TOKENIZER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "tokenizer", "fractus_bpe.json",
)


class FractusTokenizer:
    """Wrapper around a byte-level BPE tokenizer."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    @classmethod
    def gpt2_compatible(cls) -> "FractusTokenizer":
        """Load a GPT-2-compatible tokenizer from HF (same vocab, same byte-level BPE).
        This gives us a ready-to-use 50k vocab without training our own."""
        from tokenizers import Tokenizer
        tok = Tokenizer.from_pretrained("gpt2")
        return cls(tok)

    @classmethod
    def build_or_load(cls, path: Optional[str] = None) -> "FractusTokenizer":
        """Load from path if exists, else use GPT-2 compatible."""
        path = path or DEFAULT_TOKENIZER_PATH
        if os.path.exists(path):
            tok = Tokenizer.from_file(path)
            return cls(tok)
        return cls.gpt2_compatible()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [enc.ids for enc in self.tokenizer.encode_batch(texts)]

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def save(self, path: Optional[str] = None):
        path = path or DEFAULT_TOKENIZER_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tokenizer.save(path)
