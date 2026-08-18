#!/usr/bin/env python3
"""
Shard the full Fractus corpus (4.23B tokens) into 4 equal parts for multi-GPU training.
Produces: data/shard_gpu0.pt ... data/shard_gpu3.pt
"""
import torch, os, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/full_corpus.pt")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--n-shards", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading {args.src}...")
    tokens = torch.load(args.src, map_location="cpu", weights_only=False)
    print(f"Total tokens: {len(tokens):,}")

    n = len(tokens)
    shard_size = n // args.n_shards
    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(args.n_shards):
        start = i * shard_size
        end = n if i == args.n_shards - 1 else (i + 1) * shard_size
        shard = tokens[start:end].clone()
        path = os.path.join(args.out_dir, f"shard_gpu{i}.pt")
        torch.save(shard, path)
        size_gb = os.path.getsize(path) / (1024 ** 3)
        print(f"GPU {i}: {len(shard):,} tokens → {path} ({size_gb:.2f} GB)")
        del shard

    print("ALL_SHARDS_READY")

if __name__ == "__main__":
    main()
