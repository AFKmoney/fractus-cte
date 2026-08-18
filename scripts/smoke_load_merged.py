#!/usr/bin/env python3
"""Smoke-test: load HF stage2 merge into ContinuousThoughtEngine and run one tick_chunk.

Usage:
  python scripts/smoke_load_merged.py [path_to_pt]
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fractus.continuous_engine import ContinuousThoughtEngine

ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "checkpoints" / "FRACTUS_1B_STAGE2_MERGED.pt"
print("load", ckpt)
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
sd = ck.get("model_state", ck)
clean = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
eng = ContinuousThoughtEngine(
    vocab_size=50257, d_model=1280, n_layers=16, n_heads=20, d_head=64,
    n_levels=2, n_oscillators=16, coupling_rank=8, n_experts=128, top_k=2,
    expert_d_ff=2048, siren_rank=64,
)
own = eng.state_dict()
n = 0
for k, v in clean.items():
    if k in own and own[k].shape == v.shape:
        own[k] = v
        n += 1
eng.load_state_dict(own, strict=False)
eng.eval()
eng.reset_thought(1)
x = torch.randint(0, 50257, (1, 16))
with torch.no_grad():
    logits = eng.tick_chunk(x)
print("OK loaded", n, "tensors; tick_chunk out", tuple(logits.shape))
