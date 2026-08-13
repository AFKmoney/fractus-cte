#!/usr/bin/env python
"""Build the Fractus training corpus from ALL datasets in a source folder.

Consumes:
  - <src>/datasets/*.pt        → already tokenized, concatenated as-is
  - <src>/*/*.jsonl            → tokenized on the fly (streaming, memory-bounded)

This replaces the tiny inline builder in deploy_gpu.sh, which only took
5 jsonl files × 200 entries per dir — i.e. it ignored ~99% of the jsonl
data (cognitive_skills alone is 91 files × ~40MB ≈ 3.6GB).

Streaming: each jsonl file is read line-by-line and tokenized in entry-
batches (default 2000), so a multi-GB jsonl never sits fully in RAM.

Usage:
    python scripts/build_corpus.py --src data/hf_datasets --out data/training_corpus.pt
"""
import argparse, os, sys, glob, json, time, gzip
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from fractus.tokenizer import FractusTokenizer


def extract_text(entry: dict) -> str:
    """Pull raw text out of common JSONL schemas (zero hard-coded dirs)."""
    if "messages" in entry:                              # chat format
        return " ".join(m.get("content", "") for m in entry["messages"])
    if "instruction" in entry:                           # alpaca / neuro-paradigm format
        return (entry.get("instruction", "") + " "
                + entry.get("input", "") + " "
                + entry.get("response", "") + " "       # neuro_paradigms uses 'response'
                + entry.get("output", ""))
    if "prompt" in entry:                                # completion format
        return entry.get("prompt", "") + " " + entry.get("completion", "")
    if "text" in entry:                                  # plain text
        return entry["text"]
    return ""


def main():
    ap = argparse.ArgumentParser(description="Build Fractus training corpus")
    ap.add_argument("--src", default="data/hf_datasets",
                    help="folder downloaded by snapshot_download")
    ap.add_argument("--out", default="data/training_corpus.pt")
    ap.add_argument("--cap", type=int, default=1_000_000_000,
                    help="max tokens to keep (default 1B — memory-safe on a 32GB box)")
    ap.add_argument("--text-batch", type=int, default=2000,
                    help="entries tokenized per batch (memory bound)")
    ap.add_argument("--min-text-len", type=int, default=20)
    args = ap.parse_args()

    tok = FractusTokenizer.gpt2_compatible()
    chunks = []          # list of 1D int tensors
    total = 0
    t0 = time.time()

    # ── 1. Pre-tokenized .pt files ──────────────────────────────────────
    pt_files = sorted(glob.glob(os.path.join(args.src, "datasets", "*.pt")))
    print(f"=== {len(pt_files)} pre-tokenized .pt files ===", flush=True)
    for f in pt_files:
        try:
            t = torch.load(f, weights_only=False)
            if t.dim() != 1:
                t = t.reshape(-1)
            chunks.append(t.to(torch.int64))
            total += len(t)
            print(f"  {os.path.basename(f):<40} {len(t):>14,}", flush=True)
        except Exception as e:
            print(f"  SKIP {f}: {e}", flush=True)

    # ── 2. All .jsonl / .jsonl.gz anywhere under src (recursive) ────────
    jsonl_files = sorted(
        glob.glob(os.path.join(args.src, "**", "*.jsonl"), recursive=True) +
        glob.glob(os.path.join(args.src, "**", "*.jsonl.gz"), recursive=True))
    print(f"\n=== {len(jsonl_files)} .jsonl/.jsonl.gz files (streaming tokenize) ===",
          flush=True)
    for jf in jsonl_files:
        batch, file_tokens = [], 0
        opener = gzip.open if jf.endswith(".gz") else open
        with opener(jf, "rt", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    text = extract_text(json.loads(line))
                except Exception:
                    continue
                if text and len(text) > args.min_text_len:
                    batch.append(text)
                if len(batch) >= args.text_batch:
                    toks = torch.tensor(tok.encode("\n\n".join(batch)),
                                        dtype=torch.int64)
                    chunks.append(toks)
                    file_tokens += len(toks)
                    total += len(toks)
                    batch = []
        if batch:
            toks = torch.tensor(tok.encode("\n\n".join(batch)), dtype=torch.int64)
            chunks.append(toks)
            file_tokens += len(toks)
            total += len(toks)
        if file_tokens:
            print(f"  {os.path.relpath(jf, args.src):<55} {file_tokens:>12,}", flush=True)

    print(f"\nTotal available: {total:,} tokens (gathered in {time.time()-t0:.0f}s)", flush=True)

    # ── 3. Concatenate + shuffle + cap ──────────────────────────────────
    mega = torch.cat(chunks)
    n = len(mega)
    cap = min(n, args.cap)
    g = torch.Generator().manual_seed(42)
    if n <= 300_000_000:
        perm = torch.randperm(n, generator=g)            # full shuffle, fits in RAM
        mega = mega[perm]
    else:
        # Uniform sample of `cap` tokens. With-replacement when n > cap, but
        # the dup rate is cap/n (small when data is plentiful) — fine for
        # pretraining and avoids a multi-GB full permutation index.
        idx = torch.randint(0, n, (cap,), generator=g)
        mega = mega[idx]
    mega = mega.to(torch.int32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(mega, args.out)
    print(f"Saved {args.out}: {len(mega):,} tokens "
          f"({os.path.getsize(args.out)/1e6:.0f}MB)", flush=True)


if __name__ == "__main__":
    main()
