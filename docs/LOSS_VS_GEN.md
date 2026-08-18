# Why Train Loss Was Not Accurate vs Generation

**Date:** 2026-08-16 23:31 UTC

## Diagnosis

Teacher-forced CE (train metric) and free-run generation measure different things.

| Metric | GPU1-ish value | Meaning |
|--------|----------------|---------|
| Train TF CE (warm continuous) | ~1.3-1.5 | Next-token prediction with ground-truth history + carried thought state |
| Cold TF CE (reset state) | ~7-40 | Same objective but without warm dynamical state |
| Free-run AR CE vs ground truth | huge / match~0 | Model leaves data manifold when feeding its own tokens |

So a loss of 1.5 was **accurate for teacher forcing**, not for generation.

## Fix in training

Scheduled sampling (sequential, VRAM-safe):

1. Pass A: teacher-forced CE + LB (always)
2. Pass B (30% of steps): mix 20% model-sampled tokens into inputs, CE again against true targets

Logged now:

- `tf=` / `ema_tf=` teacher-forced
- `ss=` / `ema_ss=` scheduled-sampling pass (gen-closer)

Script: `fast4gpu_stage2_ss.py`

## Live (after switch)

GPU1 example: tf~1.45 while ss~2.9 — gap is visible and optimized.

