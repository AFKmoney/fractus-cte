# Fractus Operability — Mid-Training Open-Heart Surgery

**Date:** 2026-08-16 19:03 UTC

## Claim

Fractus can be **operated on during training** without discarding learned weights.
The brain (.pt) stays; the training objective and dynamical routing can be rewired; digestion continues from the same token offset.

This is not a full retrain. It is live surgery.

## Surgeries performed in this run

### Surgery A — Routing (stage 1 fix)

| Bug | Fix | Weights kept? |
|-----|-----|---------------|
| Kuramoto under torch.no_grad in tick_chunk_core | Gradients enabled on phase clock | Yes |
| lb_loss computed then detached / not in loss | CE + 0.02 * lb | Yes |
| Hard von Mises gates (temp=1) | gate temperature 2.5 | Yes |
| Near-zero omega diversity | controlled omega expand if std < 0.08 | Yes |

### Surgery B — Sequence stage (stage 2)

| Before | After |
|--------|-------|
| CE on **last position only** (1 target / 128 tokens) | **Dense CE** on all positions (B*C targets) |
| Favours local lexical attractors | Forces token-to-token chaining |

Code change in ContinuousThoughtEngine.tick_chunk_train:

    # before: last_logits = output_head(h[:, -1, :])
    # after:  logits = output_head(h)   # (B, C, vocab)

Trainer: fast4gpu_stage2.py — resumes exact start_token from logs, keeps LB + soft gates.

## Why this is possible

1. **Separation of brain and body** — weights are tensors; objective and routing are code.
2. **Compatible state_dict** — same shapes across surgeries; load strict=False only for batch-shaped buffers.
3. **Manifest / log resume** — start_token preserved so shards are not rewound to zero.
4. **Composable checkpoints** — 4 GPU shards can still mean-merge after surgery.

## Live metrics after stage-2 surgery

| GPU | Tokens | CE (dense) | lb |
|-----|--------|------------|-----|
| 0 | 177,932,800 | 17.9 | 13.253 |
| 1 | 187,750,400 | 4.2 | 14.027 |
| 2 | 188,556,800 | 6.7 | 14.019 |
| 3 | 185,536,000 | 17.6 | 14.027 |

Loss at stage-2 start can differ in scale from last-position-only CE (different objective).
What matters: descent continues, lb stays active (~14), GPUs stay full, no weight wipe.

## Relation to generation collapse

Mid-training generation showed word-level repetition (Colorado, Fate, Population loops).
Diagnosis: stage-1 lexical attractors; routing locks to same expert each tick.
Stage-2 dense CE is the mid-training intervention aimed at sequence chaining without resetting digestion.

## Operational rule

If a dynamical bottleneck is found (dead experts, frozen clock, sparse target):

1. Save / keep current .pt
2. Patch body (engine / trainer)
3. Reload weights
4. Resume at recorded token offset
5. Document the surgery

Do not throw away multi-day digestion for a routing or objective bug.

*Fractus is operable. This log is proof from production, not a design slide.*
