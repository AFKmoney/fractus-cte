# Fractus Discovery Log — Bugs, Optimizations, Emergent Features

**Updated:** 2026-08-16 18:28 UTC

This document records what we found *while* running Fractus-1B — things that go beyond the original design notes. Empirical, not marketing.

---

## 1. Critical training bugs found (and fixed)

### 1.1 Kuramoto was not learning during training

**Symptom:** `omega` stayed near init (±0.05), phase order parameter r ≈ 0.01–0.03, soft dynamics.

**Root cause:** In `CTEBlock.tick_chunk_core`, Kuramoto integration ran under `torch.no_grad()`:



Comment in code even said "clock, not learned". So CE loss never reached `omega` / coupling.

**Fix:** Remove `no_grad` around Kuramoto; keep phase *state* detached for carry, keep differentiable `theta` for MoE routing so parameters receive gradients.

**Status:** Fixed in `fractus/continuous_engine.py` (surgery 2026-08-16).

### 1.2 Load-balance loss was computed then thrown away

**Symptom:** Offline probe showed ~70% experts dead (90+/128), only 2–3 experts active per block.

**Root causes:**
1. `tick_chunk_train` did `total_lb + lb.detach()` — no gradient through LB
2. `fast4gpu.py` optimized **only** cross-entropy — never added `lb_loss` to the loss

**Fix:**
- Keep LB in the graph
- `tick_chunk_train` returns `(logits, lb_loss)`
- Surgery trainer: `loss = CE + 0.02 * lb`

**Status:** Fixed; live `lb≈14` on all GPUs after resume.

### 1.3 Probe false alarm: "all phases are zero"

**Symptom:** A probe script reported all Kuramoto phases at 0.0.

**Root cause:** Script called `reset_thought()` before measuring — which zeros state buffers.

**Reality in checkpoints:** phases nonzero, std ≈ 1.8 across all 16 blocks × 4 GPUs.

**Lesson:** Never diagnose dynamical state after an intentional reset.

---

## 2. Optimizations discovered in production

| Optimization | What we learned |
|--------------|-----------------|
| 4-GPU independent shards + mean-merge | Works; unified `.pt` generates; different lexical attractors than single shard |
| Resume from exact token offset | Manifest-driven `start_token` preserves progress across pod reboot |
| Gate temperature ↑ (1.0 → 2.5) | Softens von Mises routing; more experts can enter the top-k mix |
| Omega scale ×4 on resume | Restores phase-rate diversity without wiping weights |
| `tick_vec` multimodal path | Vision patches can drive CTE without touching token embedding |
| CPU eyes prototype (CIFAR) | Small CTE+PatchEmbed learns real images offline while 1B trains on GPU |

---

## 3. Features that emerged beyond the original plan

### 3.1 Operable / open-heart model
Weights (`.pt`) + body (`fractus/` code) are separable. We can:
- merge brains
- change routing temperature
- inject LB pressure
- add vision front-end
without a full retrain from zero.

### 3.2 Infinite-ish checkpoint fusion (same architecture)
Compatible checkpoints can be mean-merged and trained again:

```
train → merge → train → merge → ...
```

Constraint: same shapes (d_model, layers, experts, etc.). Divergent merges can soup skills; shard-merge is the proven path.

### 3.3 Mid-training generation behavior
At loss ~33→25, generation pipeline works but outputs **word-level repetition collapse** (real tokens stuck in loops: "Colorado", "Population", "Fate", "thinks"…). Documents that Fractus emits lexical tokens before coherent sentences.

### 3.4 Parallel modality track
Text 1B digestion and vision prototype can run in parallel (GPU text + CPU eyes) without stopping the main run.

### 3.5 Routing pathology as first-class debug target
Expert-hit histograms + phase order parameter `r` are necessary metrics. Loss alone hides "model learns with 3 experts".

---

## 4. Live metrics after routing surgery (resume)

Resume offsets preserved from pre-crash run (~176–187M tokens/GPU).

| GPU | Resume start | Snapshot tokens | CE loss | lb |
|-----|--------------|-----------------|---------|-----|
| 0 | 176,281,600 | 176,614,400 | 36.5 | 14.026 |
| 1 | 186,137,600 | 186,444,800 | 12.4 | 14.024 |
| 2 | 186,854,400 | 187,212,800 | 34.7 | 14.026 |
| 3 | 183,833,600 | 184,166,400 | 41.0 | 14.027 |

Post-surgery CE can spike briefly then fall (routing distribution shift). GPU1 recovered fastest into the teens.

---

## 5. What this means for the Fractus thesis

Fractus is not only "another 1B trained on shards". The run forced discovery of:

1. **Silent non-learning** of the phase clock under `no_grad`
2. **Silent expert death** without LB in the loss
3. **Composable checkpoints** as a workflow
4. **Operability** (surgery without discarding digestion)

The architecture does more than the first README described — because production training exposed the dynamical bottlenecks.

---

*Keep this log updated when new probes, merges, or surgeries land.*
