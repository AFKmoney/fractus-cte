#!/usr/bin/env python
"""GPU training script for Fractus 1B with progressive growth.

Loads the best CPU-trained checkpoint (palier 3, d=768) and grows it to 1B,
then trains on GPU with all optimizations active.

Usage (on a GPU machine):
    python scripts/train_1b_gpu.py \\
        --checkpoint checkpoints/fractus_palier3.pt \\
        --tokens 1760000000 \\
        --batch-size 4 \\
        --seq-len 64 \\
        --lr 1e-4 \\
        --accumulation-steps 4 \\
        --bf16

Expected on RTX 3090 (24GB):
    - Forward+backward per token: ~0.5ms → ~2000 tok/s
    - With sparse MoE (2/128): ~3000 tok/s effective
    - With PGSU (4/16 layers): ~1.5x more → ~4500 tok/s
    - 1.76B tokens at 4000 tok/s ≈ 5 days

With progressive growth warm start:
    - Palier 4 starts from palier 3 weights (d=768 trained)
    - Needs ~1/4 of Chinchilla to converge → ~440M tokens
    - 440M at 4000 tok/s ≈ 30 hours → ~1.5 days
"""
import argparse, os, sys, time, math, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.grow import grow_cte
from fractus.tokenizer import FractusTokenizer

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "quality_corpus.pt")

# 1B target config (white paper config K).
# n_layers=16 is required to reach ~1.05B params (verified: 1,048,631,458).
TARGET_1B = dict(
    d_model=1280, n_heads=20, d_head=64, n_levels=2,
    n_oscillators=16, coupling_rank=8,
    n_experts=128, top_k=2, expert_d_ff=2048, siren_rank=64,
    n_layers=16,
)


def load_checkpoint(engine, ckpt_path):
    """Load weights from a checkpoint, ignoring buffer mismatches."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    model_sd = ckpt["model_state"]
    own_sd = engine.state_dict()
    for key, val in model_sd.items():
        if key in own_sd and own_sd[key].shape == val.shape:
            own_sd[key] = val
    engine.load_state_dict(own_sd)
    cfg = ckpt.get("config", {})
    print(f"  Loaded checkpoint: {cfg.get('palier', '?')}, "
          f"d={cfg.get('d_model', '?')}, E={cfg.get('n_experts', '?')}", flush=True)
    return engine


def train_1b_gpu(engine, tokens, n_tokens, lr, batch_size, seq_len,
                 accumulation_steps, use_bf16, pgsu_active, log_every=500):
    """Train the 1B model on GPU with all optimizations.

    Optimizations active:
    - tick_chunk_train: head on last position only (seq_len x less head FLOPs)
    - Sparse MoE low-rank: only top-2/128 experts computed (64x less MoE work)
    - Gradient accumulation: fewer optimizer steps
    - bf16 AMP: 2x on all matmuls, halved memory
    - PGSU: 4/16 layers active per step (if enabled)
    """
    device = next(engine.parameters()).device
    vocab = engine.vocab_size
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    # SGD w/ momentum: 1 state/param (4GB) vs AdamW's 2 (8GB) — saves ~4GB,
    # letting B=4 fit on 32GB. SGD was the project's optimizer (1345 tok/s,
    # 37% faster than AdamW). lr is scaled up ~10x vs Adam defaults.
    opt = torch.optim.SGD(engine.parameters(), lr=max(lr * 10, 1e-3),
                          momentum=0.9, weight_decay=0.01)

    # PGSU setup.
    pgsu = None
    if pgsu_active:
        try:
            from fractus1B.pgsu import PGSU
            # Note: PGSU works on Fractus1B (16 blocks). For the CTE (single MoE),
            # PGSU is a no-op. This is here for when we switch to Fractus1B.
            print("  PGSU: available for Fractus1B (not used on CTE)", flush=True)
        except Exception:
            pass

    engine.train()
    # B independent streams, each carrying its own continuous thought state.
    # Batching B chunks/step saturates the GPU (was 1 chunk/step = ~50% util).
    B = max(1, batch_size)
    engine.reset_thought(batch_size=B)

    t0 = time.time()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    step_idx = 0
    opt.zero_grad()

    n = tokens.numel()
    shard = n // B                                  # each stream walks its own region
    tokens_per_stream = max(1, min(n_tokens // B, shard - seq_len - 1))
    n_steps = max(1, tokens_per_stream // seq_len)
    positions = [i * shard for i in range(B)]       # current pos of each stream

    for step in range(n_steps):
        chunks = torch.stack([tokens[p:p + seq_len] for p in positions]).to(device)
        targets = torch.tensor([tokens[p + seq_len] for p in positions], device=device)

        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=dtype):
                last_logits = engine.tick_chunk_train(chunks)
                loss = F.cross_entropy(last_logits, targets) / accumulation_steps
        else:
            last_logits = engine.tick_chunk_train(chunks)
            loss = F.cross_entropy(last_logits, targets) / accumulation_steps

        loss.backward()

        total_loss += loss.item() * accumulation_steps
        total_correct += (last_logits.argmax(dim=-1) == targets).sum().item()
        total_n += B
        step_idx += 1
        positions = [p + seq_len for p in positions]   # advance each stream

        if step_idx % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

        if step_idx % log_every == 0:
            processed = step_idx * seq_len * B
            elapsed = time.time() - t0
            rate = processed / max(elapsed, 1)
            avg = total_loss / max(total_n, 1)
            acc = total_correct / max(total_n, 1)
            ppl = math.exp(min(avg, 20))
            mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"  {processed:>10,}/{n_tokens:,} loss={avg:.3f} ppl={ppl:.1f} "
                  f"acc={acc:.3f} {rate:.0f} tok/s "
                  f"GPU_mem={mem_gb:.1f}GB (B={B})", flush=True)

    # Final remainder.
    if step_idx % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
        opt.step()

    elapsed = time.time() - t0
    avg_loss = total_loss / max(total_n, 1)
    ppl = math.exp(min(avg_loss, 20))
    done_tok = step_idx * seq_len * B
    print(f"\n  DONE: loss={avg_loss:.3f} ppl={ppl:.1f} "
          f"({done_tok:,} tokens in {elapsed/3600:.1f}h, "
          f"{done_tok / max(elapsed,1):.0f} tok/s)", flush=True)
    return engine


def evaluate(engine, tokens, n_eval=1000, seq_len=64):
    """Quick eval: perplexity on holdout."""
    engine.eval()
    device = next(engine.parameters()).device
    holdout = tokens[-n_eval:]
    total_nll, n = 0.0, 0
    engine.reset_thought(batch_size=1)
    for s in range(0, min(len(holdout) - seq_len - 1, n_eval), seq_len):
        chunk = holdout[s:s + seq_len].unsqueeze(0).to(device)
        target = holdout[s + seq_len].to(device)
        with torch.no_grad():
            logits = engine.tick_chunk_train(chunk)
            nll = F.cross_entropy(logits, target.unsqueeze(0))
        total_nll += nll.item()
        n += 1
    avg = total_nll / max(n, 1)
    return avg, math.exp(min(avg, 20))


def generate_sample(engine, tok, prompt_text, n_tokens=60):
    """Greedy decode from prompt."""
    engine.eval()
    engine.reset_thought(batch_size=1)
    device = next(engine.parameters()).device
    ids = tok.encode(prompt_text)
    for t in ids:
        engine.tick(torch.tensor([t], device=device))
    cur = torch.tensor([ids[-1]], device=device)
    generated = list(ids)
    for _ in range(n_tokens):
        with torch.no_grad():
            logits, _ = engine.tick(cur)
        nxt = int(logits.argmax(-1).item())
        generated.append(nxt)
        cur = torch.tensor([nxt], device=device)
    return tok.decode(generated)


def main():
    ap = argparse.ArgumentParser(description="Fractus 1B GPU Training")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="CPU-trained checkpoint to grow from (e.g. fractus_palier3.pt)")
    ap.add_argument("--tokens", type=int, default=440_000_000,
                    help="training tokens (default: 440M = ~1/4 Chinchilla for warm start)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accumulation-steps", type=int, default=4)
    ap.add_argument("--bf16", action="store_true", default=True,
                    help="enable bf16 mixed precision (default: on)")
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")
    ap.add_argument("--pgsu", action="store_true", help="enable PGSU (Fractus1B only)")
    ap.add_argument("--corpus", type=str, default=CORPUS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-interval", type=int, default=50000,
                    help="evaluate PPL every N tokens")
    ap.add_argument("--save-interval", type=int, default=100000,
                    help="save checkpoint every N tokens")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Fractus 1B GPU Training ===", flush=True)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)
        print(f"bf16: {args.bf16}", flush=True)
    else:
        print("WARNING: no GPU detected — training will be extremely slow!", flush=True)

    torch.set_num_threads(os.cpu_count() or 6)

    # Load corpus.
    print(f"Loading corpus: {args.corpus}", flush=True)
    tokens = torch.load(args.corpus, weights_only=False).to(torch.int64)
    print(f"Corpus: {len(tokens):,} tokens", flush=True)

    # Build engine — either grow from checkpoint or build fresh 1B.
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nLoading checkpoint: {args.checkpoint}", flush=True)
        # Build a model matching the checkpoint config, load, then grow.
        ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
        cfg = ckpt["config"]
        engine = ContinuousThoughtEngine(
            vocab_size=50257,
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            d_head=cfg.get("d_head", 64), n_levels=2,
            n_layers=cfg.get("n_layers", 1),
            n_oscillators=8, coupling_rank=4,
            n_experts=cfg["n_experts"], top_k=2,
            expert_d_ff=cfg["expert_d_ff"],
            siren_rank=cfg["siren_rank"])
        engine = load_checkpoint(engine, args.checkpoint)

        # Grow to 1B target.
        print(f"\nGrowing to 1B target: d={TARGET_1B['d_model']}, "
              f"E={TARGET_1B['n_experts']}", flush=True)
        engine = grow_cte(engine, TARGET_1B)
        print(f"  Grown: d={engine.d_model}, E={engine.blocks[0].moe.n_experts}, "
              f"params={sum(p.numel() for p in engine.parameters()):,}", flush=True)
    else:
        print("\nBuilding 1B from scratch (no checkpoint)", flush=True)
        engine = ContinuousThoughtEngine(vocab_size=50257, **TARGET_1B)
        print(f"  params={sum(p.numel() for p in engine.parameters()):,}", flush=True)

    engine = engine.to(device)

    # Eval before training.
    nll_before, ppl_before = evaluate(engine, tokens)
    print(f"\nBefore: NLL={nll_before:.3f} PPL={ppl_before:.1f}", flush=True)

    # Train.
    print(f"\nTraining: {args.tokens:,} tokens, lr={args.lr}, "
          f"accum={args.accumulation_steps}, bf16={args.bf16}", flush=True)
    engine = train_1b_gpu(
        engine, tokens, args.tokens, lr=args.lr,
        batch_size=args.batch_size, seq_len=args.seq_len,
        accumulation_steps=args.accumulation_steps,
        use_bf16=args.bf16, pgsu_active=args.pgsu)

    # Eval after.
    nll_after, ppl_after = evaluate(engine, tokens)
    print(f"After:  NLL={nll_after:.3f} PPL={ppl_after:.1f}", flush=True)

    # Generation test.
    tok = FractusTokenizer.gpt2_compatible()
    print(f"\n=== Generation Test ===", flush=True)
    for prompt in ["The function", "def fractus", "import torch", "Hello, my name is"]:
        text = generate_sample(engine, tok, prompt, n_tokens=60)
        print(f'  "{text}"', flush=True)

    # Save.
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "checkpoints", "fractus_1b_gpu.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        "model_state": engine.state_dict(),
        "config": {**TARGET_1B, "method": "gpu_progressive_growth"},
        "params": sum(p.numel() for p in engine.parameters()),
        "ppl": ppl_after,
    }, ckpt_path)
    print(f"\nSaved: {ckpt_path} ({os.path.getsize(ckpt_path)/1e9:.2f}GB)", flush=True)

    # Upload to HF.
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo="checkpoints/fractus_1b_gpu.pt",
                repo_id="thefinalboss/Fractus-1B", repo_type="model")
            print("Uploaded to HuggingFace: thefinalboss/Fractus-1B", flush=True)
        except Exception as e:
            print(f"HF upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
