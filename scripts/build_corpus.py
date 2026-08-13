#!/usr/bin/env python
"""Build the Fractus training corpus from ALL datasets in a source folder.

Consumes:
  - <src>/datasets/*.pt        → already tokenized, concatenated as-is
  - <src>/*/*.jsonl(.gz)       → tokenized on the fly, IN PARALLEL

Parallel: jsonl files are tokenized across N worker processes (default 32),
each writing its tokens to a temp .pt to avoid huge IPC. On a many-core box
this turns a multi-hour serial tokenization into minutes. Pre-tokenized .pt
files are concatenated serially (they're already tokens).

Usage:
    python scripts/build_corpus.py --src data/hf_datasets --out data/training_corpus.pt
    python scripts/build_corpus.py --src ... --workers 64 --cap 3000000000
"""
import argparse, os, sys, glob, json, time, gzip, tempfile, hashlib
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from fractus.tokenizer import FractusTokenizer


def extract_text(entry: dict) -> str:
    """Pull raw text out of common JSONL schemas (zero hard-coded dirs)."""
    if "messages" in entry:
        return " ".join(m.get("content", "") for m in entry["messages"])
    if "instruction" in entry:
        return (entry.get("instruction", "") + " "
                + entry.get("input", "") + " "
                + entry.get("response", "") + " "
                + entry.get("output", ""))
    if "prompt" in entry:
        return entry.get("prompt", "") + " " + entry.get("completion", "")
    if "text" in entry:
        return entry["text"]
    if "content" in entry and isinstance(entry["content"], str):
        return entry["content"]
    return ""


# ── Per-worker state ──────────────────────────────────────────────────────
_TOK = None
_TMPDIR = None

def _worker_init(tmpdir):
    global _TMPDIR
    _TMPDIR = tmpdir
    os.makedirs(tmpdir, exist_ok=True)

def _get_tok():
    global _TOK
    if _TOK is None:
        _TOK = FractusTokenizer.gpt2_compatible()
    return _TOK


def _tokenize_one(args):
    """Tokenize a single jsonl/jsonl.gz file → temp .pt. Returns (tmp_path, n_tokens, relpath)."""
    jf, src, text_batch, min_text_len = args
    tok = _get_tok()
    opener = gzip.open if jf.endswith(".gz") else open
    batch = []
    local = []
    file_tokens = 0
    try:
        with opener(jf, "rt", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    text = extract_text(json.loads(line))
                except Exception:
                    continue
                if text and len(text) > min_text_len:
                    batch.append(text)
                if len(batch) >= text_batch:
                    t = torch.tensor(tok.encode("\n\n".join(batch)), dtype=torch.int64)
                    local.append(t); file_tokens += len(t); batch = []
        if batch:
            t = torch.tensor(tok.encode("\n\n".join(batch)), dtype=torch.int64)
            local.append(t); file_tokens += len(t)
    except Exception as e:
        return (None, 0, os.path.relpath(jf, src), f"ERR {str(e)[:60]}")
    if not local:
        return (None, 0, os.path.relpath(jf, src), "empty")
    out = torch.cat(local)
    h = hashlib.md5(jf.encode()).hexdigest()[:10]
    tmp = os.path.join(_TMPDIR, f"tok_{os.getpid()}_{h}.pt")
    torch.save(out, tmp)
    return (tmp, file_tokens, os.path.relpath(jf, src), None)


def main():
    ap = argparse.ArgumentParser(description="Build Fractus training corpus (parallel)")
    ap.add_argument("--src", default="data/hf_datasets")
    ap.add_argument("--out", default="data/training_corpus.pt")
    ap.add_argument("--cap", type=int, default=1_000_000_000)
    ap.add_argument("--text-batch", type=int, default=2000)
    ap.add_argument("--min-text-len", type=int, default=20)
    ap.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 4)))
    args = ap.parse_args()

    t0 = time.time()
    chunks = []   # list of 1D int64 tensors (paths to temp files we load later)
    temp_paths = []
    total = 0

    # ── 1. Pre-tokenized .pt files (serial — they're already tokens) ─────
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

    # ── 2. jsonl / jsonl.gz — tokenized IN PARALLEL ─────────────────────
    jsonl_files = sorted(
        glob.glob(os.path.join(args.src, "**", "*.jsonl"), recursive=True) +
        glob.glob(os.path.join(args.src, "**", "*.jsonl.gz"), recursive=True))
    print(f"\n=== {len(jsonl_files)} .jsonl/.jsonl.gz — tokenizing on "
          f"{args.workers} workers ===", flush=True)
    tmpdir = tempfile.mkdtemp(prefix="fctok_")
    # Pre-init the tokenizer in THIS process so the forked workers inherit it
    # (avoids 64x per-worker GPT-2 init/download that serializes the pool).
    _get_tok()
    print(f"  (tokenizer pre-warmed, forking {args.workers} workers)", flush=True)
    work = [(jf, args.src, args.text_batch, args.min_text_len) for jf in jsonl_files]
    done = 0
    with mp.Pool(args.workers, initializer=_worker_init, initargs=(tmpdir,)) as pool:
        for tmp, ntok, rel, err in pool.imap_unordered(_tokenize_one, work, chunksize=1):
            done += 1
            if err:
                if done % 50 == 0 or err.startswith("ERR"):
                    print(f"  [{done}/{len(jsonl_files)}] {rel}: {err}", flush=True)
                continue
            temp_paths.append(tmp)
            total += ntok
            if done % 25 == 0:
                print(f"  [{done}/{len(jsonl_files)}] {rel:<50} {ntok:>10,}  "
                      f"(total {total:,}, {total//4//1_000_000}M tok)", flush=True)
    print(f"\n  jsonl tokenized: {len(temp_paths)} files, total now "
          f"{total:,} tokens ({total//4//1_000_000}M) in {time.time()-t0:.0f}s", flush=True)

    # Load the parallel temp tensors and concat everything.
    for tp in temp_paths:
        try:
            chunks.append(torch.load(tp, weights_only=False))
        except Exception:
            pass

    # ── 3. Concatenate + shuffle + cap ──────────────────────────────────
    print("concatenating + shuffling...", flush=True)
    mega = torch.cat(chunks)
    # free chunk list + temp files
    del chunks
    for tp in temp_paths:
        try: os.remove(tp)
        except OSError: pass
    try: os.rmdir(tmpdir)
    except OSError: pass

    n = len(mega)
    cap = min(n, args.cap)
    g = torch.Generator().manual_seed(42)
    if n <= 300_000_000:
        mega = mega[torch.randperm(n, generator=g)]
    else:
        mega = mega[torch.randint(0, n, (cap,), generator=g)]
    mega = mega.to(torch.int32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(mega, args.out)
    print(f"Saved {args.out}: {len(mega):,} tokens "
          f"({os.path.getsize(args.out)/1e6:.0f}MB) in {time.time()-t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
