#!/usr/bin/env python
"""Progressive growth training: build small → train → grow → train → ... → large.

Instead of training a large model from scratch (impossible on CPU), this script
grows the model palier by palier. Each palier inherits the previous weights via
zero-padding, then trains briefly. The model never starts from random.

Usage:
    # Default: 4 paliers on CPU, then stop (palier 5 = 1B needs GPU).
    python scripts/train_progressive.py

    # Custom: specify paliers to run.
    python scripts/train_progressive.py --paliers 0,1

    # Tiny smoke test.
    python scripts/train_progressive.py --paliers 0 --tokens-per-palier 5000
"""
import argparse, os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.train.online import OnlineTrainer
from fractus.grow import grow_cte

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "quality_corpus.pt")
# Allow override via env (set by deploy_gpu.sh) or --corpus flag (resolved in main).
CORPUS = os.environ.get("FRACTUS_CORPUS", CORPUS)

# The growth ladder. Each palier is bigger than the last.
# d_model grows ~2x, n_heads keeps d_head=64, experts grow, rank grows.
PALIERS = [
    # (d_model, n_heads, n_experts, siren_rank, expert_d_ff, tokens, lr)
    dict(d_model=128,  n_heads=2,  n_experts=4,   siren_rank=32, expert_d_ff=128,  tokens=2_000_000, lr=1e-3),
    dict(d_model=256,  n_heads=4,  n_experts=8,   siren_rank=32, expert_d_ff=256,  tokens=1_500_000, lr=5e-4),
    dict(d_model=512,  n_heads=8,  n_experts=16,  siren_rank=64, expert_d_ff=512,  tokens=1_000_000, lr=3e-4),
    dict(d_model=768,  n_heads=12, n_experts=32,  siren_rank=64, expert_d_ff=768,  tokens=500_000,   lr=2e-4),
    dict(d_model=1280, n_heads=20, n_experts=128, siren_rank=64, expert_d_ff=2048, tokens=0,         lr=1e-4),  # GPU only
]


def train_palier(engine, tokens, n_tokens, lr, palier_name, accumulation_steps=8):
    """Train one palier using the fast OnlineTrainer (chunked, head-last).

    Trains in SEGMENTS with periodic logging so we see progress.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"TRAINING {palier_name}", flush=True)
    print(f"  d_model={engine.d_model}, experts={engine.blocks[0].moe.n_experts}, "
          f"rank={engine.blocks[0].moe.expert_rank}, "
          f"params={sum(p.numel() for p in engine.parameters()):,}", flush=True)
    print(f"  tokens={n_tokens:,}, lr={lr}", flush=True)
    print(f"{'='*60}", flush=True)

    if n_tokens == 0:
        print("  (skipped — 0 tokens, GPU only)", flush=True)
        return engine

    torch.set_num_threads(os.cpu_count() or 6)
    chunk_len = 32
    trainer = OnlineTrainer(engine, lr=lr, accumulation_steps=accumulation_steps)

    # Train in ONE pass over n_tokens, with inline logging every ~50k tokens.
    train_tokens = tokens[:n_tokens]
    log_interval = max(50_000 // chunk_len, 1)  # log every ~50k tokens worth of chunks

    t0 = time.time()
    import torch.nn.functional as F
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    chunk_idx = 0
    accum = accumulation_steps
    trainer.optimizer.zero_grad()

    for start in range(0, len(train_tokens) - chunk_len - 1, chunk_len):
        chunk = train_tokens[start:start + chunk_len].unsqueeze(0)
        target = train_tokens[start + chunk_len]
        last_logits = engine.tick_chunk_train(chunk)
        loss = F.cross_entropy(last_logits, target.unsqueeze(0)) / accum
        loss.backward()

        total_loss += loss.item() * accum
        pred = last_logits.argmax(dim=-1)
        total_correct += (pred == target.unsqueeze(0)).sum().item()
        total_n += 1
        chunk_idx += 1

        if chunk_idx % accum == 0:
            torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            trainer.step_count += 1

        if chunk_idx % log_interval == 0:
            processed = chunk_idx * chunk_len
            elapsed = time.time() - t0
            rate = processed / max(elapsed, 1)
            avg = total_loss / max(total_n, 1)
            acc = total_correct / max(total_n, 1)
            ppl = math.exp(min(avg, 20))
            print(f"  {processed:>8,}/{n_tokens:,} loss={avg:.3f} "
                  f"ppl={ppl:.1f} acc={acc:.3f} {rate:.0f} tok/s", flush=True)

    # Final remainder step.
    if chunk_idx % accum != 0:
        torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()

    elapsed = time.time() - t0
    avg_loss = total_loss / max(total_n, 1)
    ppl = math.exp(min(avg_loss, 20))
    print(f"\n  {palier_name} DONE: loss={avg_loss:.3f} ppl={ppl:.1f} "
          f"({chunk_idx * chunk_len:,} tokens in {elapsed/60:.1f}min, "
          f"{chunk_idx * chunk_len / max(elapsed,1):.0f} tok/s)", flush=True)

    return engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paliers", type=str, default="0,1,2,3",
                    help="comma-separated palier indices to run (default: 0,1,2,3)")
    ap.add_argument("--tokens-per-palier", type=int, default=None,
                    help="override tokens per palier (for quick tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--accumulation-steps", type=int, default=8,
                    help="gradient accumulation steps (fewer optimizer steps = faster)")
    ap.add_argument("--compile", action="store_true",
                    help="enable torch.compile on tick_chunk_train (reduce-overhead mode)")
    ap.add_argument("--corpus", type=str, default=CORPUS,
                    help="path to tokenized corpus .pt (default: env FRACTUS_CORPUS or data/quality_corpus.pt)")
    args = ap.parse_args()

    palier_indices = [int(x) for x in args.paliers.split(",")]
    torch.manual_seed(args.seed)

    print("=== Fractus Progressive Growth ===", flush=True)
    print(f"Paliers: {palier_indices}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Accumulation steps: {args.accumulation_steps}", flush=True)
    print(f"Compile: {args.compile}", flush=True)

    # Load corpus.
    tokens = torch.load(args.corpus, weights_only=False).to(torch.int64)
    print(f"Corpus: {len(tokens):,} tokens", flush=True)

    engine = None
    for idx in palier_indices:
        config = PALIERS[idx]
        palier_name = f"Palier {idx}"

        # Check for a checkpoint from the PREVIOUS palier.
        prev_ckpt = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints", f"fractus_palier{idx - 1}.pt")

        if engine is None and os.path.exists(prev_ckpt):
            # Resume from previous palier checkpoint.
            print(f"\n--- Loading Palier {idx - 1} checkpoint: {prev_ckpt} ---", flush=True)
            ckpt = torch.load(prev_ckpt, weights_only=False, map_location="cpu")
            prev_config = ckpt.get("config", {})
            engine = ContinuousThoughtEngine(
                vocab_size=50257,
                d_model=prev_config.get("d_model", 128),
                n_heads=prev_config.get("n_heads", 2),
                d_head=prev_config.get("d_head", 64),
                n_levels=2, n_oscillators=8, coupling_rank=4,
                n_experts=prev_config.get("n_experts", 4),
                top_k=2,
                expert_d_ff=prev_config.get("expert_d_ff", 128),
                siren_rank=prev_config.get("siren_rank", 32))
            # Load weights, ignoring buffer size mismatches (kuramoto_phases etc).
            model_sd = ckpt["model_state"]
            own_sd = engine.state_dict()
            for key, val in model_sd.items():
                if key in own_sd and own_sd[key].shape == val.shape:
                    own_sd[key] = val
            engine.load_state_dict(own_sd)
            print(f"  Loaded: d={engine.d_model}, E={engine.blocks[0].moe.n_experts}, "
                  f"params={sum(p.numel() for p in engine.parameters()):,}", flush=True)

        if engine is None:
            # Palier 0: build from scratch (no checkpoint found).
            print(f"\n--- Building {palier_name} from scratch ---", flush=True)
            engine = ContinuousThoughtEngine(
                vocab_size=50257, d_model=config["d_model"],
                n_heads=config["n_heads"], d_head=64, n_levels=2,
                n_oscillators=8, coupling_rank=4,
                n_experts=config["n_experts"], top_k=2,
                expert_d_ff=config["expert_d_ff"],
                siren_rank=config["siren_rank"])
        elif engine.d_model < config["d_model"] or engine.blocks[0].moe.n_experts < config["n_experts"]:
            # Grow from previous palier (in-memory or just-loaded checkpoint).
            print(f"\n--- Growing to {palier_name} ---", flush=True)
            grow_config = dict(
                d_model=config["d_model"],
                n_heads=config["n_heads"],
                d_head=64,
                n_experts=config["n_experts"],
                siren_rank=config["siren_rank"],
                expert_d_ff=config["expert_d_ff"],
            )
            engine = grow_cte(engine, grow_config)
            print(f"  Grown: d={engine.d_model}, E={engine.blocks[0].moe.n_experts}, "
                  f"params={sum(p.numel() for p in engine.parameters()):,}", flush=True)
        else:
            print(f"\n--- {palier_name} already at target config, training in place ---", flush=True)
            print(f"  Grown: d={engine.d_model}, E={engine.blocks[0].moe.n_experts}, "
                  f"params={sum(p.numel() for p in engine.parameters()):,}", flush=True)

        # torch.compile (optional — reduces Python overhead on repeated calls).
        if args.compile:
            print(f"  Compiling tick_chunk_train (reduce-overhead)...", flush=True)
            engine.tick_chunk_train = torch.compile(
                engine.tick_chunk_train, mode="reduce-overhead")

        # Train this palier.
        n_tokens = args.tokens_per_palier or config["tokens"]
        engine = train_palier(engine, tokens, n_tokens, config["lr"], palier_name,
                              accumulation_steps=args.accumulation_steps)

        # Save checkpoint.
        ckpt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints", f"fractus_palier{idx}.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({
            "model_state": engine.state_dict(),
            "config": {
                "d_model": engine.d_model,
                "n_experts": engine.blocks[0].moe.n_experts,
                "siren_rank": engine.blocks[0].moe.expert_rank,
                "expert_d_ff": engine.blocks[0].moe.d_ff,
                "n_heads": engine.blocks[0].attn.n_heads,
                "d_head": engine.blocks[0].attn.d_head,
                "palier": idx,
            },
            "params": sum(p.numel() for p in engine.parameters()),
        }, ckpt_path)
        print(f"  Saved: {ckpt_path} ({os.path.getsize(ckpt_path)/1e6:.0f}MB)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Progressive growth complete.", flush=True)
    print(f"Final model: d={engine.d_model}, E={engine.blocks[0].moe.n_experts}, "
          f"params={sum(p.numel() for p in engine.parameters()):,}", flush=True)
    if engine.d_model < 1280:
        print(f"Next: run palier {max(palier_indices)+1} to continue growing.", flush=True)
    else:
        print(f"Reached 1B target! Ready for GPU fine-tuning.", flush=True)


if __name__ == "__main__":
    main()
