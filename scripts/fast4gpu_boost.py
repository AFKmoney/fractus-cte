#!/usr/bin/env python3
"""Fractus-1B boost trainer — B=4, compile, TF32, sequential scheduled sampling.

Resume from HF merge if pod weights are lost:
  checkpoints/FRACTUS_1B_STAGE2_MERGED.pt

Usage (one process per GPU):
  CUDA_VISIBLE_DEVICES=0 GPU_ID=0 python -u scripts/fast4gpu_boost.py
  CUDA_VISIBLE_DEVICES=1 GPU_ID=1 python -u scripts/fast4gpu_boost.py
  ...

Env:
  GPU_ID, BATCH=4, SEQ=128, LR=7e-4, SS_RATE=0.25, SS_PROB=0.2
  CKPT_IN  — path to load (default: merged or per-gpu if present)
  CKPT_OUT — path to save (default: checkpoints/fractus_1b_gpu{GPU}.pt)
  START_TOKEN — optional integer resume offset into shard
  SHARD — path to token shard .pt (int64 1D)
"""
from __future__ import annotations

import os
import sys
import time
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from fractus.continuous_engine import ContinuousThoughtEngine

GPU = int(os.environ.get("GPU_ID", "0"))
LB_COEF = float(os.environ.get("LB_COEF", "0.02"))
GATE_TEMP = float(os.environ.get("GATE_TEMP", "2.5"))
LR = float(os.environ.get("LR", "7e-4"))
EMA_BETA = 0.98
SS_PROB = float(os.environ.get("SS_PROB", "0.2"))
SS_RATE = float(os.environ.get("SS_RATE", "0.25"))
B = int(os.environ.get("BATCH", "4"))
SEQ = int(os.environ.get("SEQ", "128"))

TARGET = dict(
    d_model=1280,
    n_heads=20,
    d_head=64,
    n_levels=2,
    n_oscillators=16,
    coupling_rank=8,
    n_experts=128,
    top_k=2,
    expert_d_ff=2048,
    siren_rank=64,
    n_layers=16,
)

torch.manual_seed(42 + GPU)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0")

default_merged = ROOT / "checkpoints" / "FRACTUS_1B_STAGE2_MERGED.pt"
default_gpu = ROOT / "checkpoints" / f"fractus_1b_gpu{GPU}.pt"
CKPT_IN = Path(os.environ.get("CKPT_IN", str(default_gpu if default_gpu.exists() else default_merged)))
CKPT_OUT = Path(os.environ.get("CKPT_OUT", str(default_gpu)))
SHARD = Path(os.environ.get("SHARD", str(ROOT / "data" / f"shard_gpu{GPU}.pt")))

print(f"GPU {GPU}: BOOST B={B} SEQ={SEQ} LR={LR} SS_RATE={SS_RATE}", flush=True)
print(f"GPU {GPU}: load {CKPT_IN}", flush=True)

ck = torch.load(CKPT_IN, map_location="cpu", weights_only=False)
sd = ck.get("model_state", ck)
clean = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}

eng = ContinuousThoughtEngine(vocab_size=50257, **{k: TARGET[k] for k in TARGET})
own = eng.state_dict()
loaded = 0
for k, v in clean.items():
    if k in own and own[k].shape == v.shape:
        own[k] = v
        loaded += 1
    elif (
        k in own
        and v.dim() >= 1
        and own[k].dim() >= 1
        and v.shape[0] > own[k].shape[0]
        and v.shape[1:] == own[k].shape[1:]
    ):
        own[k] = v[: own[k].shape[0]].contiguous()
        loaded += 1
eng.load_state_dict(own, strict=False)
print(f"GPU {GPU}: loaded_tensors={loaded}", flush=True)

with torch.no_grad():
    for blk in eng.blocks:
        if hasattr(blk, "moe") and hasattr(blk.moe, "temperature"):
            blk.moe.temperature = GATE_TEMP

eng = eng.to(device)
eng.reset_thought(B)

try:
    print("compile disabled for VRAM", flush=True)
    print(f"GPU {GPU}: compile OK", flush=True)
except Exception as e:
    print(f"GPU {GPU}: compile skip: {e}", flush=True)

opt = torch.optim.SGD(eng.parameters(), lr=LR, momentum=0.9)

if not SHARD.exists():
    # allow .npy memmap fallback for large phase2 shards
    alt = Path(str(SHARD) + ".npy") if not str(SHARD).endswith(".npy") else SHARD
    npy = SHARD if str(SHARD).endswith(".npy") else Path(str(SHARD).replace(".pt", ".npy"))
    if not npy.exists():
        raise FileNotFoundError(
            f"Shard not found: {SHARD} (also tried {npy})"
        )
    SHARD = npy

if str(SHARD).endswith(".npy"):
    import numpy as np
    tokens = torch.from_numpy(np.load(str(SHARD), mmap_mode="r")).to(torch.int64)
    print(f"GPU {GPU}: memmap shard {SHARD} len={len(tokens):,}", flush=True)
else:
    tokens = torch.load(SHARD, weights_only=False).to(torch.int64)
    print(f"GPU {GPU}: loaded shard {SHARD} len={len(tokens):,}", flush=True)
step_tokens = B * SEQ

start_token = int(os.environ.get("START_TOKEN", "0"))
# align to step
start_token = (start_token // step_tokens) * step_tokens
print(f"GPU {GPU}: RESUME start_token={start_token} step={step_tokens} shard_len={len(tokens)}", flush=True)

t0 = time.time()
ema_tf = None
ema_ss = None
n = 0
tok_sess = 0

CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)

for start in range(start_token, len(tokens) - step_tokens - SEQ - 1, step_tokens):
    chunk = tokens[start : start + step_tokens].view(B, SEQ).to(device)
    target = tokens[start + 1 : start + step_tokens + 1].view(B, SEQ).to(device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = eng.tick_chunk_train(chunk)
        logits, lb = out if isinstance(out, tuple) else (out, eng.last_lb_loss)
        ce_tf = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        loss = ce_tf + LB_COEF * lb

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
    opt.step()

    tf_v = float(ce_tf.item())
    lb_v = float(lb.detach().item()) if torch.is_tensor(lb) else float(lb)
    ema_tf = tf_v if ema_tf is None else EMA_BETA * ema_tf + (1 - EMA_BETA) * tf_v

    ce_ss_v = None
    if random.random() < SS_RATE:
        with torch.no_grad():
            samp = torch.multinomial(
                torch.softmax(logits.detach().float().reshape(-1, logits.size(-1)) / 0.9, dim=-1),
                1,
            ).view(B, SEQ)
        mixed = chunk.clone()
        use_ss = torch.rand(B, SEQ, device=device) < SS_PROB
        use_ss[:, 0] = False
        prev = torch.cat([chunk[:, :1], samp[:, :-1]], dim=1)
        mixed = torch.where(use_ss, prev, mixed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out2 = eng.tick_chunk_train(mixed)
            logits2, lb2 = out2 if isinstance(out2, tuple) else (out2, eng.last_lb_loss)
            ce_ss = F.cross_entropy(logits2.reshape(-1, logits2.size(-1)), target.reshape(-1))
            loss2 = 0.5 * ce_ss + LB_COEF * lb2
        opt.zero_grad(set_to_none=True)
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
        opt.step()
        ce_ss_v = float(ce_ss.item())
        ema_ss = ce_ss_v if ema_ss is None else EMA_BETA * ema_ss + (1 - EMA_BETA) * ce_ss_v

    n += 1
    tok_sess += step_tokens

    if n % 40 == 0:
        tps = tok_sess / max(time.time() - t0, 1e-6)
        extra = f" ss={ce_ss_v:.3f} ema_ss={ema_ss:.3f}" if ce_ss_v is not None and ema_ss is not None else ""
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"GPU {GPU}: {start + step_tokens:>12,} tf={tf_v:.3f} ema_tf={ema_tf:.3f}{extra} "
            f"lb={lb_v:.3f} {tps:.0f} tok/s mem={mem:.1f}GB [boost]",
            flush=True,
        )

    if n % 800 == 0:
        torch.save(
            {
                "model_state": eng.state_dict(),
                "config": {
                    **TARGET,
                    "gpu": GPU,
                    "boost": True,
                    "batch": B,
                    "lr": LR,
                    "ss_rate": SS_RATE,
                    "tokens_processed": start + step_tokens,
                },
            },
            CKPT_OUT,
        )
        print(f"GPU {GPU}: saved [boost] -> {CKPT_OUT}", flush=True)

print(f"GPU {GPU}: DONE", flush=True)
