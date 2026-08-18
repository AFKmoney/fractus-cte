# Boost B=4 Recovery Recipe (HF)

**Updated:** 2026-08-17 03:10 UTC

## Purpose

If the QuickPod volume is lost, resume training from Hugging Face using:

- weights: `checkpoints/FRACTUS_1B_STAGE2_MERGED.pt` (last merge uploaded **2026-08-16 19:10 UTC / 12:10 PM PDT**)
- trainer: `scripts/fast4gpu_boost.py`
- code: `fractus/` package on this repo

This is the **boost configuration that was running when the pod died** (B=4, ~940–960 tok/s/GPU on 5090).

## Boost settings (canonical)

| Knob | Value |
|------|--------|
| BATCH | **4** |
| SEQ | 128 |
| LR | **7e-4** |
| momentum SGD | 0.9 |
| torch.compile | reduce-overhead |
| TF32 | on |
| dense CE | all positions via `tick_chunk_train` |
| LB coef | 0.02 |
| gate temperature | 2.5 |
| SS_RATE | 0.25 (fraction of steps with second pass) |
| SS_PROB | 0.20 (fraction of positions replaced in SS pass) |
| SS implementation | **sequential** (no dual graph — avoids OOM) |

## On a new machine

```bash
# 1. clone / download repo
huggingface-cli download thefinalboss/fractus-cte --local-dir ./fractus-cte
cd fractus-cte

# 2. ensure data shards exist (re-tokenize if needed)
# data/shard_gpu0.pt ... shard_gpu3.pt  (int64 1D token streams)

# 3. smoke load merge (CPU ok)
python scripts/smoke_load_merged.py checkpoints/FRACTUS_1B_STAGE2_MERGED.pt

# 4. launch 4 GPU boost from merge
CUDA_VISIBLE_DEVICES=0 GPU_ID=0 CKPT_IN=checkpoints/FRACTUS_1B_STAGE2_MERGED.pt \
  python -u scripts/fast4gpu_boost.py

CUDA_VISIBLE_DEVICES=1 GPU_ID=1 CKPT_IN=checkpoints/FRACTUS_1B_STAGE2_MERGED.pt \
  python -u scripts/fast4gpu_boost.py

# same for GPU 2 and 3
```

Optional: `START_TOKEN=0` or a known offset if you recovered manifests.

## After each save — push to HF

```bash
huggingface-cli upload thefinalboss/fractus-cte checkpoints/fractus_1b_gpu0.pt checkpoints/fractus_1b_gpu0.pt
# repeat gpu1-3; mean-merge periodically
```

**Lesson from outage:** weights must leave the rental host continuously.

## What this does / does not recover

| Recovered from HF merge | Not recovered (pod-only unless volume returns) |
|-------------------------|--------------------------------------------------|
| Stage2 dense-CE digestion up to merge time | Last ~hours of boost after merge |
| Architecture + operable train path | Exact 203M token offsets on GPU1 |
| Boost recipe identical to production | Live thought-state of the crashed run |

You do **not** start from random. You start from **stage2 merged brain** + boost loop.

## Related

- `docs/TRUSTED_LOSS.md` — which loss to trust
- `docs/LOSS_VS_GEN.md` — TF vs free-run
- `docs/COMPOSABILITY_AND_SURGERY.md` — merge / surgery
- `docs/MASTER_RUN_LOG.md` — full chronology
