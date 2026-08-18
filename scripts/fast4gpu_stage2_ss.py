import torch, sys, os, time, json, random
sys.path.insert(0, "/workspace/fractus-cte")
os.chdir("/workspace/fractus-cte")
from fractus.continuous_engine import ContinuousThoughtEngine
import torch.nn.functional as F

GPU = int(os.environ.get("GPU_ID", "0"))
LB_COEF = float(os.environ.get("LB_COEF", "0.02"))
GATE_TEMP = float(os.environ.get("GATE_TEMP", "2.5"))
LR = float(os.environ.get("LR", "5e-4"))
EMA_BETA = 0.98
SS_PROB = float(os.environ.get("SS_PROB", "0.2"))
SS_RATE = float(os.environ.get("SS_RATE", "0.3"))
torch.manual_seed(42 + GPU)
device = torch.device("cuda:0")

TARGET = dict(d_model=1280, n_heads=20, d_head=64, n_levels=2, n_oscillators=16,
    coupling_rank=8, n_experts=128, top_k=2, expert_d_ff=2048, siren_rank=64, n_layers=16)

print(f"GPU {GPU}: SS sequential LR={LR} SS_PROB={SS_PROB}", flush=True)
ck = torch.load(f"checkpoints/fractus_1b_gpu{GPU}.pt", map_location="cpu", weights_only=False)
sd = ck.get("model_state", ck)
clean = {(k[10:] if k.startswith("_orig_mod.") else k): v for k,v in sd.items()}
eng = ContinuousThoughtEngine(vocab_size=50257, **{k: TARGET[k] for k in TARGET})
own = eng.state_dict()
for k,v in clean.items():
    if k in own and own[k].shape == v.shape:
        own[k] = v
    elif k in own and v.dim()>=1 and own[k].dim()>=1 and v.shape[0]>own[k].shape[0] and v.shape[1:]==own[k].shape[1:]:
        own[k] = v[:own[k].shape[0]].contiguous()
eng.load_state_dict(own, strict=False)
with torch.no_grad():
    for blk in eng.blocks:
        blk.moe.temperature = GATE_TEMP
eng = eng.to(device)
eng.reset_thought(2)
opt = torch.optim.SGD(eng.parameters(), lr=LR, momentum=0.9)
tokens = torch.load(f"data/shard_gpu{GPU}.pt", weights_only=False).to(torch.int64)
B, seq_len = 2, 128
step_tokens = B * seq_len
man = json.load(open("/workspace/RESUME_MANIFEST_SS.json"))
start_token = int(man["gpus"][str(GPU)]["start_token"])
print(f"GPU {GPU}: RESUME {start_token}", flush=True)
t0=time.time(); ema_tf=None; ema_ss=None; n=0; tok_sess=0
for start in range(start_token, len(tokens)-step_tokens-seq_len-1, step_tokens):
    chunk = tokens[start:start+step_tokens].view(B, seq_len).to(device)
    target = tokens[start+1:start+step_tokens+1].view(B, seq_len).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = eng.tick_chunk_train(chunk)
        logits, lb = out if isinstance(out, tuple) else (out, eng.last_lb_loss)
        ce_tf = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        loss = ce_tf + LB_COEF * lb
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0); opt.step()
    tf_v=float(ce_tf.item()); lb_v=float(lb.detach().item()) if torch.is_tensor(lb) else float(lb)
    ema_tf = tf_v if ema_tf is None else EMA_BETA*ema_tf+(1-EMA_BETA)*tf_v
    ce_ss_v=None
    if random.random() < SS_RATE:
        with torch.no_grad():
            samp = torch.multinomial(torch.softmax(logits.detach().float().reshape(-1, logits.size(-1))/0.9, -1), 1).view(B, seq_len)
        mixed = chunk.clone()
        use_ss = torch.rand(B, seq_len, device=device) < SS_PROB
        use_ss[:,0]=False
        prev = torch.cat([chunk[:,:1], samp[:,:-1]], 1)
        mixed = torch.where(use_ss, prev, mixed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out2 = eng.tick_chunk_train(mixed)
            logits2, lb2 = out2 if isinstance(out2, tuple) else (out2, eng.last_lb_loss)
            ce_ss = F.cross_entropy(logits2.reshape(-1, logits2.size(-1)), target.reshape(-1))
            loss2 = 0.5*ce_ss + LB_COEF*lb2
        opt.zero_grad(); loss2.backward()
        torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0); opt.step()
        ce_ss_v=float(ce_ss.item())
        ema_ss = ce_ss_v if ema_ss is None else EMA_BETA*ema_ss+(1-EMA_BETA)*ce_ss_v
    n+=1; tok_sess+=step_tokens
    if n%50==0:
        tps=tok_sess/max(time.time()-t0,1e-6)
        extra=f" ss={ce_ss_v:.3f} ema_ss={ema_ss:.3f}" if ce_ss_v is not None and ema_ss is not None else ""
        print(f"GPU {GPU}: {start+step_tokens:>12,} tf={tf_v:.3f} ema_tf={ema_tf:.3f}{extra} lb={lb_v:.3f} {tps:.0f} tok/s [ss]", flush=True)
    if n%1000==0:
        torch.save({"model_state": eng.state_dict(), "config": {**TARGET, "gpu": GPU, "ss": True}}, f"checkpoints/fractus_1b_gpu{GPU}.pt")
        print(f"GPU {GPU}: saved [ss]", flush=True)
print("DONE", flush=True)
