#!/usr/bin/env python
"""Fast sequential tokenizer — no multiprocessing, no fork issues.
Uses the Rust tokenizer's native speed. Writes one corpus file at the end.

Usage: python scripts/fast_tokenize.py --src data/hf_datasets --out data/full_corpus.pt
"""
import torch, glob, json, gzip, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fractus.tokenizer import FractusTokenizer


def extract(entry):
    if "messages" in e: return " ".join(m.get("content", "") for m in entry["messages"])
    if "instruction" in entry:
        return entry.get("instruction", "") + " " + entry.get("input", "") + " " + \
               entry.get("response", "") + " " + entry.get("output", "")
    if "text" in entry: return entry["text"]
    if "content" in entry and isinstance(entry["content"], str): return entry["content"]
    return ""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hf_datasets")
    ap.add_argument("--out", default="data/full_corpus.pt")
    ap.add_argument("--cap", type=int, default=4_000_000_000)
    args = ap.parse_args()

    tok = FractusTokenizer.gpt2_compatible()
    src = args.src
    all_ids = []  # list of 1D tensors
    total = 0
    t0 = time.time()

    # 1. Pre-tokenized .pt files (instant)
    for f in sorted(glob.glob(src + "/datasets/*.pt")):
        try:
            t = torch.load(f, weights_only=False)
            if t.dim() != 1: t = t.reshape(-1)
            all_ids.append(t.to(torch.int64))
            total += len(t)
        except: pass
    print(f".pt: {total:,} tokens ({total//4//1_000_000}M) in {time.time()-t0:.0f}s", flush=True)

    # 2. JSONL + JSONL.GZ — tokenize with Rust speed
    files = sorted(
        glob.glob(src + "/**/*.jsonl", recursive=True) +
        glob.glob(src + "/**/*.jsonl.gz", recursive=True))
    print(f"Tokenizing {len(files)} jsonl/gz files...", flush=True)

    for fi, jf in enumerate(files):
        file_ids = []
        opener = gzip.open if jf.endswith(".gz") else open
        try:
            with opener(jf, "rt", encoding="utf-8", errors="ignore") as fh:
                texts = []
                for line in fh:
                    try:
                        e = json.loads(line)
                        text = ""
                        if "messages" in e:
                            text = " ".join(m.get("content", "") for m in e["messages"])
                        elif "instruction" in e:
                            text = e.get("instruction", "") + " " + e.get("response", "") + " " + e.get("output", "")
                        elif "text" in e:
                            text = e["text"]
                        elif "content" in e and isinstance(e["content"], str):
                            text = e["content"]
                        if text and len(text) > 20:
                            texts.append(text)
                    except: pass

                    # Batch encode every 500 entries or 2MB of text
                    if len(texts) >= 500 or sum(len(t) for t in texts) > 2_000_000:
                        combined = "\n\n".join(texts)
                        ids = tok.encode(combined)
                        file_ids.append(torch.tensor(ids, dtype=torch.int32))
                        texts = []

                if texts:
                    ids = tok.encode("\n\n".join(texts))
                    file_ids.append(torch.tensor(ids, dtype=torch.int32))

            if file_ids:
                fcat = torch.cat(file_ids)
                all_ids.append(fcat.to(torch.int64))
                total += len(fcat)
        except Exception as e:
            print(f"  SKIP {jf}: {e}", flush=True)
            continue

        if (fi + 1) % 25 == 0 or total - (total if fi == 0 else 0) > 50_000_000:
            elapsed = time.time() - t0
            print(f"  [{fi+1}/{len(files)}] {os.path.relpath(jf, src)[:50]:<52} "
                  f"total={total:,} ({total//4//1_000_000}M) {elapsed:.0f}s", flush=True)

    # 3. Concat + shuffle + cap + save
    print(f"Concatenating {total:,} tokens ({total//4//1_000_000}M)...", flush=True)
    mega = torch.cat(all_ids)
    del all_ids
    g = torch.Generator().manual_seed(42)
    n = len(mega)
    cap = min(n, args.cap)
    if n <= 300_000_000:
        mega = mega[torch.randperm(n, generator=g)]
    else:
        mega = mega[torch.randint(0, n, (cap,), generator=g)]
    mega = mega.to(torch.int32)
    torch.save(mega, args.out)
    print(f"SAVED: {len(mega):,} tokens ({os.path.getsize(args.out)//1_000_000}MB) "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
