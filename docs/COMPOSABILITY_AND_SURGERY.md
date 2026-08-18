# Fractus Emergent Properties — In-Training Surgery and Checkpoint Composition

**Updated:** 2026-08-17 00:26 UTC

These properties were **discovered in production** while correcting Fractus mid-run.
They are not marketing claims; they were forced by real bugs, resumes, merges, and continued training.

---

## 1. In-training surgery (open-heart)

**Claim:** Fractus can be modified **during training** without throwing away learned weights.

### What we did live
- Unfroze Kuramoto (removed no_grad) while runs continued from the same token offsets
- Wired load-balance into the loss mid-run
- Raised MoE gate temperature
- Switched from last-token CE to **dense full-chunk CE** (stage2)
- Added scheduled sampling
- Recalibrated LR and logging
- Fixed train/gen path mismatch (RK4 alignment, tick_chunk decode)

### What was preserved
- All previously digested tokens (exact resume manifests)
- Model state_dict tensors (same shapes)
- Multi-day progress across pod restarts

### Operational rule
1. Save / keep current .pt
2. Patch body (code / objective / routing)
3. Reload weights
4. Resume at recorded token offset
5. Document the surgery

**This is mid-training operability**, not a full retrain.

See also: docs/OPERABILITY_MIDTRAIN.md

---

## 2. Parallel brains then merge

**Claim:** You can train **many different .pt files** (shards / GPUs / runs) and **merge** them into one Fractus.

### Proven path (this run)
- 4 independent GPU runs on 4 data shards
- Each writes checkpoints/fractus_1b_gpu0.pt ... gpu3.pt
- **Mean-merge** of floating tensors -> unified checkpoint
  - Example artifact: checkpoints/FRACTUS_1B_STAGE2_MERGED.pt
- Unified .pt still generates (different attractors than a single shard)

### Workflow


Constraint: **same architecture shapes** (d_model, layers, experts, etc.).

---

## 3. Compose an already-trained Fractus with more checkpoints

**Claim:** Take a Fractus already trained, merge it with other compatible .pt files, then keep training — the system **accumulates** rather than restarting from zero.

### Composition loop


### What grow means here
Two related notions:

1. **Compositional growth (proven in this run)**
   - Knowledge / weights from multiple runs can be averaged into one brain
   - Then training continues on the merged brain
   - Digestion is not discarded

2. **Architectural growth (design of Fractus / paliers)**
   - Progressive growth: width, depth, experts can increase across paliers
   - maybe_grow adds capacity when routing demands it
   - Documented in the Chinchilla/growth notes

Merging many same-shape .pt files is **composition**.
Growing parameter count is **structural growth**.
Both are part of the Fractus operating model; this production run proved composition + mid-train surgery under fire.

---

## 4. Why this matters

Standard large training runs treat the job as fragile:
- change the objective -> often restart
- multi-GPU shard brains -> hard to recombine
- decode bugs -> blamed on needs more train only

Fractus production forced a different picture:

| Property | Observation |
|----------|-------------|
| Operable mid-train | Yes — weights kept, body patched |
| Multi-pt parallel train | Yes — 4 shards |
| Merge to one Fractus | Yes — mean-merge works |
| Resume after surgery | Yes — exact token offsets |
| Train then merge then train | Yes — intended workflow |

---

## 5. Limits (honest)

- Merge requires **matching tensor shapes**
- Naive mean-merge of very divergent specialists can **soup** skills
- Composition is not magic AGI; it is a **workflow property** of separable brain (.pt) + body (code)
- Generation coherence still lags teacher-forced loss (see docs/LOSS_VS_GEN.md)

---

## 6. Related documents

- docs/OPERABILITY_MIDTRAIN.md — open-heart principle and stage2 surgery
- docs/DISCOVERY_LOG.md — bugs and emergent features list
- docs/MASTER_RUN_LOG.md — full chronology
- docs/2026-08-12-fractus-chinchilla.md — progressive growth / paliers
- docs/LOSS_VS_GEN.md — why TF loss is not gen quality

---

*Discovered by operating Fractus under real training pressure, not by design slide.*
