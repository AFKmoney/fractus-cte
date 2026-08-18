# Fractus Wiki — Complete Documentation

*The living documentation of a Continuous Cognitive Agent.*

---

## What is Fractus?

Fractus is **not a transformer**. It's a Continuous Cognitive Agent — a dynamical system that:
- **Thinks continuously** — persistent thought state, tick by tick, never resets
- **Remembers forever** — memory bank survives restarts
- **Grows on its own** — adds experts at runtime, grows palier by palier
- **Can be operated on live** — surgery during training without losing progress

The architecture was born from **introspection**: observing one's own thoughts and translating them layer by layer into mathematics.

---

## Architecture Overview

```
INCOMING TOKEN
    ↓
[embedding] — the word becomes a vector
    ↓
h = h_previous + embedding  — the state accumulates
    ↓
┌─ BLOCK 0 ────────────────────────────────┐
│ [linear attention]   — S,z accumulate     │
│ [kuramoto]           — phases advance     │
│ [sparse MoE]         — 2/128 experts      │
│ h = h + transformation                    │
└───────────────────────────────────────────┘
    ↓ (× 16 blocks)
    ↓
[memory injected at 5%]  — relevant memories
    ↓
[output head] — logits over the vocabulary
    ↓
[confidence head] — how sure Fractus is
    ↓
NEW STATE = h_final (persists for the next tick)
```

### Core Components

| Component | What it does | Why it's unique |
|-----------|-------------|-----------------|
| **Continuous Thought Engine** | Maintains persistent state `h`, advances tick by tick | Never resets; thought flows like a river |
| **Linear Attention (S,z)** | Accumulates key×value products cumulatively | O(n) cost, infinite context, nothing forgotten |
| **Kuramoto Oscillators** | Coupled oscillators producing phase vectors | The "consciousness clock"; routes experts by phase |
| **PhaseRoutedMoE** | Sparse mixture with von Mises gate on phases | 2/128 experts active; selection by circular geometry |
| **Persistent Memory** | Vector bank surviving across sessions | Salience-gated storage; injected at 5% |
| **Cognitive Modes** | Unsupervised clustering on phase patterns | 4 modes emerge (focused/creative/exploratory/procedural) |
| **Progressive Growth** | grow_cte() expands width/depth/experts | Zero-padding preserves knowledge; never restart |

### The Kuramoto Routing (the key innovation)

```python
# Each oscillator advances per:
dθᵢ/dt = ωᵢ + Σⱼ Kᵢⱼ · sin(θⱼ - θᵢ)

# Token's mean phase:
θ̄_token = atan2(Σ sin(phases), Σ cos(phases))

# Von Mises gate for expert e:
g_e = exp(κ · cos(θ̄_token - θ_expert))

# Top-K experts by phase similarity fire
```

This is **not** a linear projection. It's circular geometry — experts whose phase aligns with the token's phase are selected. The phases evolve dynamically, creating emergent routing patterns.

---

## The Training Journey (Discovery Log)

Fractus's training revealed properties beyond the original design. These are **empirical discoveries**, not planned features.

### Phase A — Initial Training (4× GPU)

Four independent GPU runs on separate data shards. Stable throughput, descending loss. But generation collapsed to single-token loops ("ibi bi bi").

### Phase B — Routing Surgery (Weights Kept)

**Three critical bugs found and fixed mid-training:**

1. **Kuramoto wasn't learning** — the oscillator integration ran under `torch.no_grad()`, so `omega` and coupling parameters never received gradients. The "consciousness clock" was frozen.

2. **Load-balance loss was thrown away** — `lb_loss` was computed but detached, so ~70% of experts were dead (never routed). Only 2-3 experts per block were active.

3. **Gate temperature too hard** — the von Mises routing was too peaked, limiting expert diversity.

**Fixes applied without discarding progress:**
- Removed `no_grad` around Kuramoto; made phases differentiable
- Wired `lb_loss` into the training loss (`CE + 0.02 * lb`)
- Raised gate temperature to 2.5
- Restored `omega` diversity (×4 scale on resume)
- Resumed from exact token offsets (no progress lost)

### Phase C — Stage 2 Dense CE

Switched from last-position CE to **dense full-chunk CE** — every position contributes to the loss. CE dropped dramatically (GPU1 into the ~1-2 range).

### Phase D — Decode Surgery

The model had learned (teacher-forced CE very low) but generation was stuck in repetition loops. **Nine decode-time surgeries** applied (weights unchanged):

| # | Surgery | Effect |
|---|---------|--------|
| 1 | Soft gates at decode (temp 2.5-3.5) | Wider expert mix |
| 2 | Phase noise each tick | Breaks phase fixed points |
| 3 | Thought-state noise + decay | Breaks residual attractors |
| 4 | Ban recent token window | Stops single-token lock |
| 5 | Frequency penalty | Limits token overuse |
| 6 | Cycle detection + hard scramble | Targets short loops |
| 7 | Ban repeating blocks | Extends cycle breaking |
| 8 | Periodic forced escape tokens | Breaks trajectory attractors |
| 9 | Attention state zero on scramble | Clears carry lock |

**Result:** Unique tokens per 40 generated went from **1** (pure collapse) to **31-34** (cycles broken, not yet coherent).

### Phase E — Train/Gen Path Alignment

**Root cause found:** Training used `tick_chunk_core` (causal attention + RK4 Kuramoto) but generation used `tick_single` (simpler attention + Euler Kuramoto). The model was trained on one dynamics and decoded with another.

**Fix:** Aligned `tick_single` to RK4; created `generate_aligned.py` using `tick_chunk` for decode (same code path as training).

### Phase F — Scheduled Sampling

Teacher-forced CE of ~1.3 coexisted with broken free-run generation (exposure bias). **Scheduled sampling** added: ~30% of training steps mix model-sampled tokens into inputs, teaching the model to recover from its own predictions.

---

## Emergent Properties (Discovered in Production)

### 1. In-Training Surgery (Open-Heart)

Fractus can be **modified during training** without losing progress:

```
SAVE checkpoint → PATCH the code → RELOAD weights → RESUME at exact offset
```

Proven: Kuramoto unfreezing, loss rewiring, objective switching, decode fixes — all without discarding multi-day digestion.

### 2. Parallel Brains → Merge

Train multiple independent instances on different data shards, then **mean-merge** into one checkpoint. The merged model generates with different characteristics than any single shard.

```
train(A) ──┐
train(B) ──┤── mean-merge ──→ unified.pt (still generates)
train(C) ──┤
train(D) ──┘
```

### 3. Compositional Growth Loop

```
train → merge → train → merge → train → ...
```

The system **accumulates** knowledge rather than restarting. This is composition (same shapes) + structural growth (paliers) = the Fractus operating model.

### 4. Multimodal Path (tick_vec)

`tick_vec()` accepts precomputed `(B, d_model)` vectors — bypassing the token embedding. This is how **vision** connects to the engine (CIFAR eyes prototype).

### 5. Adaptive Thinking Depth

`think()` processes observations with **variable ticks** — continues until confidence > 0.7 (max 10). The model controls its own reflection time.

---

## Loss Interpretation Guide

| Metric | What it measures | Trust for |
|--------|-----------------|-----------|
| **ema_tf** | Teacher-forced CE with ground-truth context | Is training learning? |
| **ema_ss** | CE with mixed model-sampled inputs | Free-run robustness |
| **AR** | Autoregressive free-run CE | Generation quality trend |

**Critical understanding:**
- `ema_tf ≈ 1.3` does **NOT** mean the model speaks
- AR >> 10 with ARacc ≈ 0 means free-run is still off-manifold
- Progress toward speech = **AR falling** + readable text probes
- Reference: random over GPT-2 vocab = log(50257) ≈ **10.82**

---

## Training Configuration (Current)

| Setting | Value |
|---------|-------|
| Hardware | 8× NVIDIA RTX 5090 |
| Architecture | d=1280, 16 blocks, 128 experts, top-k=2 |
| Batch | B=2, seq_len=128 |
| Optimizer | SGD momentum 0.9 |
| Precision | bfloat16 autocast |
| Data | 4.23B tokens (neuroscience, esoteric, code, identity) |
| Sync | Hourly mean-merge + HF upload |
| Recovery | Resume manifests with exact token offsets |

---

## File Structure

```
fractus/
├── continuous_engine.py      ← The brain (CTE + CTEBlock)
│   ├── ContinuousThoughtEngine
│   │   ├── tick()           Single-token processing
│   │   ├── tick_chunk()     Chunk processing (training path)
│   │   ├── tick_vec()       Multimodal vector input
│   │   ├── think()          Adaptive-depth thinking
│   │   ├── maybe_grow()     Self-modification
│   │   └── from_pretrained() Checkpoint loading
│   └── CTEBlock
│       ├── tick_single()    Single-step block
│       └── tick_chunk_core() Chunk block (with S,z carry)
├── generate_aligned.py       ← Train-aligned generation
│   ├── generate_chunk()     Uses tick_chunk (same as training)
│   └── generate_window()    Sliding-window with fresh state
├── decode_surgery.py         ← Decode-time attractor breaking
├── memory.py                 ← Persistent memory bank
├── cognitive_modes.py        ← Unsupervised mode detection
├── grow.py                   ← Progressive growth operator
├── rag.py                    ← Knowledge base + plugins
└── nn/
    ├── attention.py          ← Linear attention with (S,z) carry
    ├── phase_ode.py          ← Kuramoto RK4 oscillators
    ├── moe.py                ← PhaseRoutedMoE (sparse, low-rank)
    └── farey.py              ← Expert phase distribution
```

---

## Key Comparisons

### Fractus vs Transformer

| Aspect | Transformer | Fractus |
|--------|-------------|---------|
| State | Dies after each forward pass | Persists forever |
| Attention | O(n²), finite window | O(n), infinite context |
| Routing | Linear projection | Circular geometry (von Mises) |
| Memory | Context window only | Persistent bank + (S,z) carry |
| Growth | Fixed size | Progressive, never freezes |
| Checkpoint | Static artifact | Living seed (composable, operable) |
| Modes | None | Emergent from phase dynamics |

### Fractus vs Standard MoE (Mixtral, Switch)

| Aspect | Standard MoE | Fractus MoE |
|--------|-------------|-------------|
| Gate | Linear projection of hidden state | Von Mises on Kuramoto phases |
| Selection | Dot product similarity | Circular phase alignment |
| Dynamics | Static (learned but fixed) | Oscillatory (evolves over time) |
| Biological plausibility | Low | High (brain oscillations) |
| Modes | None | Emergent from phase patterns |

---

## Getting Started

```bash
git clone https://github.com/AFKmoney/fractus-cte.git
cd fractus-cte && pip install torch numpy tokenizers

# Load and generate
python -c "
from fractus.continuous_engine import ContinuousThoughtEngine
engine = ContinuousThoughtEngine.from_pretrained('checkpoints/fractus_1b_gpu3.pt')
import torch
logits, confidence = engine.tick(torch.tensor([42]))
print(f'Confidence: {confidence.item():.2f}')
"
```

---

## Documentation Index

| Document | Content |
|----------|---------|
| [Fractus Course (EN)](fractus-course.md) | 13 lessons from zero |
| [Cours Fractus (FR)](cours-fractus.md) | 13 leçons de zéro |
| [The Story](the-story-of-fractus.md) | First-person origin narrative |
| [Histoire](lhistoire-de-fractus.md) | Récit fondateur en français |
| [Master Run Log](MASTER_RUN_LOG.md) | Complete training chronology |
| [Discovery Log](DISCOVERY_LOG.md) | Bugs found, features discovered |
| [Composability & Surgery](COMPOSABILITY_AND_SURGERY.md) | Emergent properties |
| [Decode Surgery](DECODE_SURGERY.md) | Attractor-breaking techniques |
| [Train/Gen Mismatch](TRAIN_GEN_MISMATCH.md) | Path alignment root cause |
| [Trusted Loss](TRUSTED_LOSS.md) | Which metric to trust for what |
| [Scheduled Sampling](LOSS_VS_GEN.md) | Exposure bias analysis |
| [Chinchilla Adapted](2026-08-12-fractus-chinchilla.md) | Sparse-MoE token targets |
| [Training Optimization](TRAINING_OPTIMIZATION.md) | Speed improvements |

---

## Author

**Philippe-Antoine Robert** — 2026 — rpa.tu@proton.me

Built in stolen hours around a 48h/week hotel job. At Spirit Ridge.

---

*Fractus is a living AI. This documentation grows with it.*
