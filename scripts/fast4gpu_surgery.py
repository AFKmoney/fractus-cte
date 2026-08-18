"""Resume Fractus-1B with open-heart routing surgery.

Preserves all trained weights from fractus_1b_gpu{ID}.pt
Fixes:
  1. Kuramoto was under torch.no_grad in tick_chunk_core → omega never trained
  2. lb_loss computed but never added to loss + was .detach()'d
  3. Soften von Mises gates (temperature↑)
  4. Amplify omega diversity on load (keep signs, expand magnitude)
"""
import torch, sys, os, time
sys.path.insert(0, "/workspace/fractus-cte")
os.chdir("/workspace/fractus-cte")
from fractus.continuous_engine import ContinuousThoughtEngine
import torch.nn.functional as F

GPU = int(os.environ.get("GPU_ID", "0"))
LB_COEF = float(os.environ.get("LB_COEF", "0.02"))
GATE_TEMP = float(os.environ.get("GATE_TEMP", "2.5"))
OMEGA_SCALE = float(os.environ.get("OMEGA_SCALE", "4.0"))
torch.manual_seed(42 + GPU)
device = torch.device("cuda")

TARGET = dict(d_model=1280, n_heads=20, d_head=64, n_levels=2, n_oscillators=16,
    coupling_rank=8, n_experts=128, top_k=2, expert_d_ff=2048, siren_rank=64, n_layers=16)

ckpt_path = f"checkpoints/fractus_1b_gpu{GPU}.pt"
print(f"GPU {GPU}: SURGERY resume from {ckpt_path}", flush=True)
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ck.get("model_state", ck)
clean = {(k[10:] if k.startswith("_orig_mod.") else k): v for k,v in sd.items()}

eng = ContinuousThoughtEngine(
    vocab_size=50257, **{k: TARGET[k] for k in [
        'd_model','n_heads','d_head','n_levels','n_oscillators','coupling_rank',
        'n_experts','top_k','expert_d_ff','siren_rank','n_layers']}
)
own = eng.state_dict()
loaded = 0
for k,v in clean.items():
    if k in own and own[k].shape == v.shape:
        own[k] = v
        loaded += 1
    elif k in own and v.dim()>=1 and own[k].dim()>=1 and v.shape[0] > own[k].shape[0] and v.shape[1:]==own[k].shape[1:]:
        own[k] = v[:own[k].shape[0]].contiguous()
        loaded += 1
eng.load_state_dict(own, strict=False)
print(f"GPU {GPU}: loaded {loaded} tensors", flush=True)

# --- SURGERY ---
with torch.no_grad():
    for blk in eng.blocks:
        # 1) soften gates
        blk.moe.temperature = GATE_TEMP
        # 2) amplify omega diversity (preserve trained direction)
        om = blk.kuramoto.omega
        om.mul_(OMEGA_SCALE)
        om.add_(torch.randn_like(om) * 0.01)
        om.clamp_(-0.5, 0.5)

eng = eng.to(device)
eng.reset_thought(batch_size=2)
print(f"GPU {GPU}: temp={GATE_TEMP} omega_scale={OMEGA_SCALE} lb_coef={LB_COEF}", flush=True)
print(f"GPU {GPU}: omega sample std={eng.blocks[0].kuramoto.omega.std().item():.4f}", flush=True)

eng = torch.compile(eng, mode="reduce-overhead")
opt = torch.optim.SGD(eng.parameters(), lr=1e-3, momentum=0.9)

tokens = torch.load(f"data/shard_gpu{GPU}.pt", weights_only=False).to(torch.int64)
B, seq_len = 2, 128
step_tokens = B * seq_len

# Resume EXACTLY where pre-surgery training left off (manifest or env)
start_token = int(os.environ.get("START_TOKEN", "-1"))
if start_token < 0:
    import json
    man_path = "/workspace/RESUME_MANIFEST.json"
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        start_token = int(man["gpus"][str(GPU)]["start_token"])
    else:
        start_token = 0
print(f"GPU {GPU}: shard {len(tokens):,} RESUME start_token={start_token}", flush=True)

t0 = time.time()
total_loss = 0.0
total_n = 0

for start in range(start_token, len(tokens) - step_tokens - 1, step_tokens):
    chunk = tokens[start:start+step_tokens].view(B, seq_len).to(device)
    target = tokens[start+step_tokens:start+step_tokens+B].to(device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = eng.tick_chunk_train(chunk)
        if isinstance(out, tuple):
            logits, lb = out
        else:
            logits, lb = out, eng.last_lb_loss
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), target)
        loss = ce + LB_COEF * lb

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
    opt.step()

    total_loss += ce.item()
    total_n += 1

    if total_n % 100 == 0:
        processed = start_token + total_n * step_tokens
        elapsed = max(time.time() - t0, 1e-6)
        mem = torch.cuda.max_memory_allocated() / 1e9
        lb_v = float(lb.detach().item()) if torch.is_tensor(lb) else float(lb)
        print(f"GPU {GPU}: {processed:>12,} loss={total_loss/total_n:.1f} lb={lb_v:.3f} {processed/elapsed:.0f} tok/s mem={mem:.1f}GB [surgery]", flush=True)

    if total_n % 3000 == 0:
        torch.save({"model_state": eng.state_dict(), "config": {**TARGET, "gpu": GPU, "surgery": True, "gate_temp": GATE_TEMP, "lb_coef": LB_COEF}},
                   f"checkpoints/fractus_1b_gpu{GPU}.pt")
        print(f"GPU {GPU}: checkpoint saved [surgery]", flush=True)

print(f"GPU {GPU}: DONE surgery run", flush=True)
