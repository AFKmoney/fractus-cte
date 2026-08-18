import torch, sys, os, time, json, re
sys.path.insert(0, '/workspace/fractus-cte')
os.chdir('/workspace/fractus-cte')
from fractus.continuous_engine import ContinuousThoughtEngine
import torch.nn.functional as F

GPU = int(os.environ.get('GPU_ID', '0'))
LB_COEF = float(os.environ.get('LB_COEF', '0.02'))
GATE_TEMP = float(os.environ.get('GATE_TEMP', '2.5'))
LR = float(os.environ.get('LR', '5e-4'))  # fine phase
EMA_BETA = float(os.environ.get('EMA_BETA', '0.98'))
torch.manual_seed(42 + GPU)
device = torch.device('cuda:0')

TARGET = dict(d_model=1280, n_heads=20, d_head=64, n_levels=2, n_oscillators=16,
    coupling_rank=8, n_experts=128, top_k=2, expert_d_ff=2048, siren_rank=64, n_layers=16)

ckpt_path = f'checkpoints/fractus_1b_gpu{GPU}.pt'
print(f'GPU {GPU}: STAGE2-FINE resume {ckpt_path} LR={LR}', flush=True)
ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ck.get('model_state', ck)
clean = {(k[10:] if k.startswith('_orig_mod.') else k): v for k,v in sd.items()}

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
        om = blk.kuramoto.omega
        if float(om.detach().std()) < 0.08:
            om.mul_(4.0)
            om.add_(torch.randn_like(om) * 0.01)
            om.clamp_(-0.5, 0.5)

eng = eng.to(device)
eng.reset_thought(batch_size=2)
print(f'GPU {GPU}: temp={GATE_TEMP} lb={LB_COEF} omega_std={float(eng.blocks[0].kuramoto.omega.detach().std()):.4f}', flush=True)

eng = torch.compile(eng, mode='reduce-overhead')
opt = torch.optim.SGD(eng.parameters(), lr=LR, momentum=0.9)

tokens = torch.load(f'data/shard_gpu{GPU}.pt', weights_only=False).to(torch.int64)
B, seq_len = 2, 128
step_tokens = B * seq_len

start_token = int(os.environ.get('START_TOKEN', '-1'))
if start_token < 0:
    # prefer manifest
    man_path = '/workspace/RESUME_MANIFEST_FINE.json'
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        start_token = int(man['gpus'][str(GPU)]['start_token'])
    else:
        for logp in [f'/workspace/stage2_gpu{GPU}.log', f'/workspace/stage2_fine_gpu{GPU}.log']:
            if os.path.exists(logp):
                for line in reversed(open(logp).read().strip().splitlines()):
                    m = re.search(r'(\d[\d,]*)\s+loss=', line)
                    if m:
                        start_token = (int(m.group(1).replace(',','')) // step_tokens) * step_tokens
                        break
            if start_token >= 0:
                break
    if start_token < 0:
        start_token = 0
print(f'GPU {GPU}: RESUME start_token={start_token}', flush=True)

t0 = time.time()
ema_ce = None
ema_lb = None
total_n = 0
tokens_this_session = 0

for start in range(start_token, len(tokens) - step_tokens - seq_len - 1, step_tokens):
    chunk = tokens[start:start+step_tokens].view(B, seq_len).to(device)
    target = tokens[start+1:start+step_tokens+1].view(B, seq_len).to(device)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = eng.tick_chunk_train(chunk)
        logits, lb = out if isinstance(out, tuple) else (out, eng.last_lb_loss)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        loss = ce + LB_COEF * lb

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
    opt.step()

    ce_v = float(ce.item())
    lb_v = float(lb.detach().item()) if torch.is_tensor(lb) else float(lb)
    ema_ce = ce_v if ema_ce is None else (EMA_BETA * ema_ce + (1 - EMA_BETA) * ce_v)
    ema_lb = lb_v if ema_lb is None else (EMA_BETA * ema_lb + (1 - EMA_BETA) * lb_v)
    total_n += 1
    tokens_this_session += step_tokens

    if total_n % 50 == 0:
        processed = start + step_tokens
        elapsed = max(time.time() - t0, 1e-6)
        real_tps = tokens_this_session / elapsed
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(
            f'GPU {GPU}: {processed:>12,} ce={ce_v:.3f} ema={ema_ce:.3f} lb={lb_v:.3f} '
            f'{real_tps:.0f} tok/s mem={mem:.1f}GB [fine LR={LR}]',
            flush=True,
        )

    if total_n % 1000 == 0:
        torch.save(
            {
                'model_state': eng.state_dict(),
                'config': {
                    **TARGET,
                    'gpu': GPU,
                    'stage2_dense_ce': True,
                    'fine_phase': True,
                    'lr': LR,
                    'gate_temp': GATE_TEMP,
                    'lb_coef': LB_COEF,
                },
            },
            f'checkpoints/fractus_1b_gpu{GPU}.pt',
        )
        print(f'GPU {GPU}: checkpoint saved [fine]', flush=True)

print(f'GPU {GPU}: DONE fine phase', flush=True)
