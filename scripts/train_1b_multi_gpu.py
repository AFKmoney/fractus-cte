#!/usr/bin/env python3
"""
Fractus 1B multi-GPU training — production script (Aug 2026)

- Loads palier0 → grows to 1B
- Uses per-GPU shards of the full corpus (4.23B tokens)
- B=2, seq=128, bf16, torch.compile
- Phase-routed MoE + Continuous Thought Engine intact
- Saves checkpoints every ~3000 steps

Usage (one process per GPU):
  GPU_ID=0 CUDA_VISIBLE_DEVICES=0 python scripts/train_1b_multi_gpu.py
  GPU_ID=1 CUDA_VISIBLE_DEVICES=1 python scripts/train_1b_multi_gpu.py
  ...

Or launch all:
  for i in 0 1 2 3; do
    GPU_ID=$i CUDA_VISIBLE_DEVICES=$i setsid python -u scripts/train_1b_multi_gpu.py > logs/gpu$i.log 2>&1 &
  done
"""
import torch, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.grow import grow_cte
import torch.nn.functional as F

GPU = int(os.environ.get("GPU_ID", "0"))
torch.manual_seed(42 + GPU)
device = torch.device("cuda")

print(f"GPU {GPU}: loading palier0...", flush=True)
eng = ContinuousThoughtEngine.from_pretrained("checkpoints/fractus_palier0.pt")

TARGET = dict(
    d_model=1280, n_heads=20, d_head=64, n_levels=2, n_oscillators=16,
    coupling_rank=8, n_experts=128, top_k=2, expert_d_ff=2048,
    siren_rank=64, n_layers=16
)
eng = grow_cte(eng, TARGET).to(device)
print(f"GPU {GPU}: 1B ready ({sum(p.numel() for p in eng.parameters()):,} params)", flush=True)

B = 2
seq_len = 128
eng.reset_thought(batch_size=B)

print(f"GPU {GPU}: compiling...", flush=True)
eng = torch.compile(eng, mode="reduce-overhead")
print(f"GPU {GPU}: compiled", flush=True)

opt = torch.optim.SGD(eng.parameters(), lr=1e-3, momentum=0.9)

shard_path = f"data/shard_gpu{GPU}.pt"
if not os.path.exists(shard_path):
    raise FileNotFoundError(f"Missing {shard_path}. Run scripts/shard_corpus.py first.")

tokens = torch.load(shard_path, weights_only=False).to(torch.int64)
print(f"GPU {GPU}: shard {len(tokens):,} tokens | B={B} seq={seq_len} + compile", flush=True)

t0 = time.time()
total_loss = 0.0
total_n = 0
step_tokens = B * seq_len

for start in range(0, len(tokens) - step_tokens - 1, step_tokens):
    chunk = tokens[start:start + step_tokens].view(B, seq_len).to(device)
    target = tokens[start + step_tokens:start + step_tokens + B].to(device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = eng.tick_chunk_train(chunk)
        loss = F.cross_entropy(out.view(-1, out.size(-1)), target)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
    opt.step()

    total_loss += loss.item()
    total_n += 1

    if total_n % 100 == 0:
        processed = total_n * step_tokens
        elapsed = max(time.time() - t0, 1e-6)
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"GPU {GPU}: {processed:>12,} loss={total_loss/total_n:.1f} {processed/elapsed:.0f} tok/s mem={mem:.1f}GB", flush=True)

    if total_n % 3000 == 0:
        ckpt_path = f"checkpoints/fractus_1b_gpu{GPU}.pt"
        torch.save({
            "model_state": eng.state_dict(),
            "config": {**TARGET, "gpu": GPU, "B": B, "seq": seq_len}
        }, ckpt_path)
        print(f"GPU {GPU}: checkpoint saved → {ckpt_path}", flush=True)

print(f"GPU {GPU}: DONE full shard", flush=True)
