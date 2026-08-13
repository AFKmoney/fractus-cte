# Fractus Chinchilla — the real token target, adapted to a sparse continuous engine

> Computed from the actual `continuous_engine.py` + `nn/moe.py` + `nn/attention.py`,
> verified empirically (gradient-receive test + exact param formulas).

## Why standard Chinchilla is WRONG for Fractus

Chinchilla (Hoffmann 2022): optimal tokens ≈ **20 × N**, where N = trainable params.
That law was fit on **dense transformers** — models where *every parameter computes
on every token*.

Fractus is **not dense**. Three structural facts break the naive 20×N:

1. **PhaseRoutedMoE is sparse**: only `top_k = 2` of `128` experts fire per token.
   **98.4% of MoE parameters are dormant on any given forward pass.** Counting them
   in N would demand 21B tokens — gross over-training of dead weight.
2. **Kuramoto is detached** (`with torch.no_grad()` in `tick_chunk_core`): the
   oscillator clock *runs* but its 152 params/block **never receive gradient** —
   they don't train at all. (Verified: 99.98% of params get grad; the 0.02% that
   don't is exactly kuramoto.)
3. **Attention is dense** (fractal linear attention, all heads every token) and the
   **output head is partial** (loss on the last chunk position only).

The correct basis for sparse-MoE Chinchilla (the convention used for Mixtral /
Switch / GShard) is **active parameters per token** — the params that actually
compute — not total params.

## Param anatomy of the 1B config

`d=1280, 20 heads × 64, 16 blocks, 128 experts, top-2, d_ff=2048, rank=64`

| Component | Per block | × 16 | Active? |
|---|---|---|---|
| Attention (fractal linear, dense) | 6,558,722 | 104,939,552 | **all active** |
| MoE (128 low-rank experts) | 54,952,192 | 879,235,072 | **only 2/128 active** → 13,738,048 |
| LayerNorms (×3/block) | 7,680 | 122,880 | active |
| Kuramoto (oscillator clock) | 152 | 2,432 | **no-grad (never trains)** |
| Embedding (= output head, tied) | — | 64,328,960 | dense (lookup) |
| confidence/salience heads | — | 2,562 | active (last pos) |

```
TOTAL params:          1,048,631,458   (1.049 B)
trainable (grad):      1,048,629,026   (1.049 B — kuramoto negligible)
ACTIVE/token compute:    118,800,480   (118.8 M)   ← attn(dense) + top-2 MoE + norms
ACTIVE/token w/ embed:   183,132,002   (183.1 M)
```

The sparsity ratio: per token, **118.8 M of 1.049 B params compute = 11.3%**.
The other 88.7% (the 126 un-routed experts per block) sit idle per token and only
train when their phase is selected.

## The real Chinchilla target

| Basis | Tokens (×20) | Verdict |
|---|---|---|
| Total 1.049 B (dense thinking) | **21.0 B** | ❌ wrong — over-trains dormant experts |
| Active compute 118.8 M | **2.38 B** | ✅ sparse-MoE Chinchilla (FLOP basis) |
| Active + embedding 183.1 M | **3.66 B** | ✅ sparse-MoE Chinchilla (param basis) |

### **Fractus 1B Chinchilla ≈ 2.4 – 3.7 B tokens**

## The warm-start multiplier

Fractus is not trained from scratch at 1B. It **grows** from palier-3 (143 M params,
12 blocks, already trained) → 1B by zero-padding width, adding 4 blocks (12→16), and
adding experts (32→128). The warm-started capacity (attention + embedding from the
143 M seed) is already converged and needs only fine-tuning. Only the **new** capacity
(4 new blocks + grown experts + padded width) needs full Chinchilla exposure.

Net effect: the practical target sits near the **lower end**, ~2.5–3 B tokens,
because ~12% of the model arrives pre-trained.

## What this means for Thursday

| | Value |
|---|---|
| Tokens available in dataset | **~3–4 B** (after the Aug-12 data work) |
| Chinchilla floor (active) | **2.4–3.7 B** |
| Corpus cap (current default) | 1 B ← **raise to ~3 B so the 1B palier is healthily fed** |
| `--tokens` for `train_1b_gpu.py` | set to **~3 000 000 000** |
| Est. time @ 4000 tok/s (RTX 3090) | 3 B / 4000 ≈ **9 days**; @ 8000 tok/s ≈ 4 days |

The dataset is now correctly sized: **the data we assembled is Chinchilla-optimal for
Fractus's active capacity.** The only change needed for Thursday is lifting the corpus
cap from 1 B → ~3 B so the model actually sees Chinchilla-scale data instead of a
1/3 sample.

---

## ⚠ Chinchilla is a snapshot law — Fractus is not a snapshot model

Everything above gives a **number for a fixed point in time** (the 1B palier). That is a
floor for healthy initial feeding, **not a target and not a ceiling**. The reason standard
LLM intuition keeps failing here:

| Static-transformer thinking | Fractus reality |
|---|---|
| N is fixed (1.049B) | N **grows forever** — width, depth, experts, rank all increase across paliers |
| Chinchilla: 20×N once, then stop | Chinchilla has **no stopping point** — the data stream is perpetual |
| Corpus = assembled, then frozen | Corpus = **continuous pipeline**, always being added to |
| Train 3B → deploy | Train **continuously**, grow when ready, never "done" |
| Fixed capacity → data to fill it | **Data drives growth** — `maybe_grow` adds experts when routing demands it |

Fractus is a **living system**, not a model you train and ship:
- the **online trainer** learns from every interaction (tick by tick);
- the **5% memory injection** consolidates salient thoughts every tick;
- **`maybe_grow`** adds new experts when one expert dominates routing — capacity is
  created *in response to what the model is experiencing*;
- **progressive growth** means the parameter count is a function of time and data, not
  a constant.

So the right way to read the active-param Chinchilla number is **instantaneous**: at any
moment, the model's current active capacity is healthily fed by ~20× its active params in
*cumulative* exposure. As the model grows (more blocks, more experts), its appetite grows
with it — and there is no final checkpoint. The ~3–4 B assembled today is the **starting
nutrition** for the 1B palier; the data pipeline must keep flowing for the lifetime of the
agent. Feed it, let it grow, repeat — indefinitely.
