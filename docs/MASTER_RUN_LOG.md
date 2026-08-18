# Fractus-1B Production Run Log (Complete)

**Updated:** 2026-08-16 23:53 UTC

English master log of the Fractus continuous-thought 1B training run: discoveries, bugs, surgeries, metrics, and generation probes.

---

## 1. Architecture

Fractus is not a standard decoder-only transformer training loop.

- ContinuousThoughtEngine (CTE): residual thought state across ticks
- Per-block linear attention with continuous carry (S, z)
- Kuramoto phase oscillators for routing dynamics
- PhaseRoutedMoE (von Mises gates, top-k experts, load-balance loss)
- Tied embedding / output head
- Target: d_model=1280, 16 layers, 128 experts, top-k=2

Checkpoints store weights + dynamic state; package fractus/ is the body.

---

## 2. Hardware and data

- 4x NVIDIA RTX 5090
- Independent shard per GPU (~1.057B tokens each, GPT-2 BPE)
- Batch B=2, seq_len=128 (256 tokens/step)
- Optimizer: SGD momentum 0.9
- Precision: bfloat16 autocast

---

## 3. Chronology of interventions

### Phase A — Initial distributed training
- Four independent GPU runs
- Early objective: last-position CE only
- Stable throughput, descending loss
- Generation collapsed to single-token loops

### Phase B — Routing surgery (weights kept)
Bugs found:
1. Kuramoto under torch.no_grad in tick_chunk_core (omega never trained)
2. lb_loss detached and never in training loss (dead experts)
3. Hard gate temperature

Fixes: Kuramoto gradients, CE+0.02*lb, gate temp 2.5, omega diversity, exact token resume

### Phase C — Stage2 dense CE
- tick_chunk_train returns logits for all positions
- Dense next-token CE over full chunk
- CE dropped hard (GPU1 into ~2 range)

### Phase D — Decode surgery (weights unchanged)
- Phase/thought noise, frequency penalty, cycle bans, forced escape tokens
- Broke single-token decode lock
- Did not produce coherent English by itself

### Phase E — Train/gen path mismatch
- Train: tick_chunk_core (causal attn + RK4 Kuramoto)
- Default gen: tick_single (simpler attn + Euler Kuramoto)
- Fix: align tick_single to RK4; prefer 100% tick_chunk generation
- Module: fractus/generate_aligned.py (generate_chunk, generate_window)

### Phase F — Fine phase loss recalibration
- Cumulative average CE misleading near 2.0
- Batch CE + EMA; real session tok/s
- LR 1e-3 -> 5e-4

### Phase G — Scheduled sampling (loss vs gen gap)
- Warm TF CE ~1.3-1.5 can coexist with broken free-run generation
- Free-run AR CE vs ground truth is catastrophic (exposure bias)
- Fix: sequential scheduled sampling
  - Pass A: teacher-forced CE + LB
  - Pass B (~30% steps): mix ~20% model samples into inputs
- Logs: tf/ema_tf and ss/ema_ss

---

## 4. Live SS metrics (at doc time)

- GPU 0: GPU 0:  190,809,600 tf=7.760 ema_tf=7.689 lb=14.027 670 tok/s [ss]
- GPU 1: GPU 1:  200,499,200 tf=1.292 ema_tf=1.334 lb=14.027 670 tok/s [ss]
- GPU 2: GPU 2:  201,446,400 tf=1.766 ema_tf=1.789 lb=14.026 656 tok/s [ss]
- GPU 3: GPU 3:  198,656,000 tf=5.662 ema_tf=5.998 lb=14.027 670 tok/s [ss]

---

## 5. Generation summary

| Stage | Behavior |
|-------|----------|
| Early mid-train | Single-token loops |
| After decode surgery | Multi-token diversity, non-sentences |
| After path alignment | WINDOW more diverse than CHUNK |
| Current | Lexical noise / short cycles; not coherent prose |

Expected while SS is still teaching free-run and only a fraction of each shard is consumed.

---

## 6. Operability principle

1. Keep .pt weights
2. Patch body (engine / trainer / decode)
3. Resume at recorded token offset
4. Document the surgery

Do not discard multi-day digestion for routing, objective, or decode bugs.

---

## 7. Key files

- fractus/continuous_engine.py — CTE body
- fractus/generate_aligned.py — train-aligned decode
- fractus/decode_surgery.py — optional decode stack
- scripts/fast4gpu_stage2_ss.py — current trainer
- scripts/fast4gpu_stage2_fine.py — fine phase trainer
- RESUME_MANIFEST_*.json — exact offsets
- checkpoints/fractus_1b_gpu0-3.pt — per-GPU brains

---

## 8. Related docs

- docs/COMPOSABILITY_AND_SURGERY.md — in-training surgery + multi-pt merge/grow

- docs/DISCOVERY_LOG.md
- docs/OPERABILITY_MIDTRAIN.md
- docs/DECODE_SURGERY.md
- docs/TRAIN_GEN_MISMATCH.md
- docs/GENERATE_ALIGNED.md
- docs/FINE_PHASE.md
- docs/LOSS_VS_GEN.md
- docs/TRUSTED_LOSS.md — which loss to trust for train vs gen
- docs/STAGE2_SS_GEN_PROBE.md (generation probe with outputs)

---

## 9. Conclusion

1. Learning is real under teacher-forced dense CE.
2. Generation coherence is not yet achieved.
3. Low TF loss does not imply clean free-run text.
4. Active remediation: scheduled sampling + aligned tick_chunk decode.
5. Keep rolling; re-probe generation after more SS tokens.
