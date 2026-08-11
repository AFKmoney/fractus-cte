#!/usr/bin/env python
"""Build a quality corpus that includes Fractus's own source code.

Combines:
  - The existing communication_corpus (18.6M tokens of dialogue/text)
  - ALL Python source files from fractus/ and fractus1B/
  - All markdown docs
  - The white paper (extracted text)

The model learns its own architecture — this is the palimpseste principle:
Fractus contains its own description.
"""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from fractus.tokenizer import FractusTokenizer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect_source_files():
    """Collect all .py and .md files from the fractus packages + docs."""
    patterns = [
        os.path.join(HERE, "fractus", "**", "*.py"),
        os.path.join(HERE, "fractus1B", "**", "*.py"),
        os.path.join(HERE, "docs", "**", "*.md"),
        os.path.join(HERE, "experiments", "**", "*.py"),
        os.path.join(HERE, "scripts", "*.py"),
        os.path.join(HERE, "*.py"),
        os.path.join(HERE, "README.md"),
    ]
    files = set()
    for pattern in patterns:
        files.update(glob.glob(pattern, recursive=True))
    # Filter out __pycache__.
    files = sorted(f for f in files if "__pycache__" not in f and ".pyc" not in f)
    return files


def main():
    tok = FractusTokenizer.gpt2_compatible()
    print("=== Building Quality Corpus (with Fractus source code) ===", flush=True)

    # 1. Collect source files.
    source_files = collect_source_files()
    print(f"Source files: {len(source_files)}", flush=True)

    # Tokenize all source files.
    all_source_tokens = []
    total_chars = 0
    for fpath in source_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            total_chars += len(text)
            # Add file separator token.
            text = "\n\n# === FILE: " + os.path.relpath(fpath, HERE) + " ===\n\n" + text
            ids = tok.encode(text)
            all_source_tokens.extend(ids)
        except Exception as e:
            print(f"  skip {fpath}: {e}", flush=True)

    source_tensor = torch.tensor(all_source_tokens, dtype=torch.int32)
    print(f"Source code: {total_chars:,} chars → {len(source_tensor):,} tokens", flush=True)

    # 2. Load existing corpus.
    existing = torch.load(os.path.join(HERE, "data", "communication_corpus.pt"),
                          weights_only=False)
    print(f"Existing corpus: {len(existing):,} tokens", flush=True)

    # 3. Combine: existing + source code repeated 3x (so the model really learns its code).
    combined = torch.cat([
        existing,
        source_tensor, source_tensor, source_tensor,  # 3x for emphasis
    ])
    print(f"Combined corpus: {len(combined):,} tokens", flush=True)

    # 4. Shuffle (fixed seed).
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(combined), generator=g)
    combined = combined[perm]
    print(f"Shuffled (seed=42)", flush=True)

    # 5. Save.
    out_path = os.path.join(HERE, "data", "quality_corpus.pt")
    torch.save(combined, out_path)
    print(f"Saved: {out_path} ({os.path.getsize(out_path)/1e6:.0f}MB)", flush=True)


if __name__ == "__main__":
    main()
