# Fractus-1B Training Log (Live Run)

**Documented:** 2026-08-16 01:37 UTC

## Setup

| Item | Value |
|------|-------|
| Model | Fractus CTE 1B (Continuous Thought Engine) |
| Parameters | 1,048,631,458 |
| Hardware | 4× NVIDIA RTX 5090 |
| Parallelism | 4 independent shard processes (fast4gpu.py) |
| Batch / seq | B=2, seq=128 |
| Precision | bfloat16 autocast |
| Optimizer | SGD (lr=1e-3, momentum=0.9), grad clip 1.0 |
| Architecture | 16 layers, d_model=1280, 20 heads, 128 experts top-k=2, 16 Kuramoto oscillators/block |
| Seed | Progressive growth from fractus_palier0.pt|

## Live metrics (latest)

| GPU | Tokens seen | Loss | Throughput |
|-----|-------------|------|------------|
| 0 | 113,152,000 | **42.4** | 1012 tok/s |
| 1 | 120,396,800 | **33.1** | 1077 tok/s |
| 2 | 121,088,000 | **35.6** | 1083 tok/s |
| 3 | 121,139,200 | **36.7** | 1083 tok/s |

## Loss history (GPU 1 leader, sampled)

| Tokens | Loss |
|--------|------|
| 25,600 | 122.8 |
| 8,601,600 | 118.4 |
| 17,203,200 | 116.8 |
| 25,804,800 | 107.7 |
| 34,406,400 | 91.5 |
| 43,008,000 | 77.4 |
| 51,609,600 | 66.7 |
| 60,211,200 | 58.6 |
| 68,787,200 | 52.4 |
| 77,388,800 | 47.5 |
| 85,990,400 | 43.5 |
| 94,592,000 | 40.3 |
| 103,193,600 | 37.5 |
| 111,795,200 | 35.1 |
| 120,396,800 | 33.1 |

## Per-window descent rate (GPU 1)

| Token window | Approx. rate (loss / M tokens) |
|--------------|--------------------------------|
| 0–30M | 0.76 |
| 30–60M | 1.37 (steepest) |
| 60–90M | 0.56 |
| 90–120M | 0.29 |
| Last ~20M | ~0.27 |

Full-run average rate (GPU1): **~0.75 loss / M tokens** (122.8 → 33.2 over ~120M tokens).

Rate is **still negative** (loss keeps falling) but **decelerating** — expected mid-training behavior, not a stall.

## Generation probe (mid-training, loss ≈ 33)

Checkpoint: `fractus_1b_gpu1.pt` (leader shard).
Loaded **391 / 440** tensors into `ContinuousThoughtEngine` (train-shaped state buffers skipped/reset).

Sampling: temperature 0.9, top-k 50, max 40 new tokens, tick-by-tick CTE decode.

### Raw outputs

```
PROMPT: The meaning of life is
OUTPUT: ibibibibibibibibibibibibibibibibibibibibibibibibibibibibibibibibibibibibibibibib

PROMPT: Hello, my name is
OUTPUT: iddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddlesiddles

PROMPT: In mathematics, a continuous system
OUTPUT:  clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich clich

PROMPT: Fractus thinks
OUTPUT:  thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks thinks
```

### Interpretation

- The **generation pipeline works** (embedding → tick loop → logits → sample → decode).
- Outputs show classic **repetition collapse** at this loss level — not coherent language yet.
- For Fractus, loss is **not** assumed to map 1:1 onto GPT-style linguistic quality; coherence will be re-probed near loss 25 and 20.
- Documenting failure modes mid-run is intentional: this is an empirical log, not a marketing card.

## Multimodal note (parallel track)

- tick_vec + fractus/nn/vision.py merged for vision observations.
- CIFAR-10 eyes prototype trained on CPU (fractus_eyes_cifar_final.pt), proving image patches can drive continuous thought.
- Text 1B run was **not** interrupted for eyes work.



## Unified 4-GPU merge generation probe

**Artifact:** `FRACTUS_1B_MERGED_GENTEST.pt` (4.66 GB)  
**Method:** mean of live `fractus_1b_gpu{0,1,2,3}.pt` shard checkpoints (strip `_orig_mod.`, reset stateful buffers to batch=1)  
**Load:** 424 / 440 tensors into `ContinuousThoughtEngine`  
**Date:** 2026-08-16 (mid-run probe)

Same sampling recipe as the single-shard probe (temperature 0.9, top-k 50, max 40 new tokens, tick-by-tick decode).

### Raw outputs (unified merge)

```
PROMPT: The meaning of life is
OUTPUT:  uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe

PROMPT: Hello, my name is
OUTPUT:  Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado

PROMPT: In mathematics, a continuous system
OUTPUT:  sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails sails

PROMPT: Fractus thinks
OUTPUT:  Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population
```

### Notes

- Pipeline works on the **averaged** 4-GPU brain, not only on the leader shard.
- Still classic **repetition collapse** at this loss band — not coherent language.
- Token identity differs from GPU1-only probe (`ibib`/`iddles`/`clich`/`thinks` vs `uphe`/`Colorado`/`sails`/`Population`), which is expected: mean weights ≠ best shard.
- Re-probe after further loss drop (targets ~25 and ~20).




## Unified merge probe — loss ≈ 30 (2026-08-16 05:28 UTC)

**Live shard losses at merge time**

| GPU | Tokens | Loss |
|-----|--------|------|
| 0 | 127.9M | 38.6 |
| 1 | 135.7M | **30.0** |
| 2 | 136.4M | 32.3 |
| 3 | 136.0M | 33.5 |

**Artifact:** `FRACTUS_1B_MERGED_LIVE.pt` (4.66 GB) — mean of 4 live shard checkpoints  
**Load:** 424 / 440 tensors  

### Raw generation outputs (unified)

```
PROMPT: The meaning of life is
OUTPUT:  uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe ...

PROMPT: Hello, my name is
OUTPUT:  Colorado Colorado Colorado Colorado Colorado Colorado ...

PROMPT: In mathematics, a continuous system
OUTPUT:  Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate ...

PROMPT: Fractus thinks
OUTPUT:  Population Population Population Population Population ...

PROMPT: Once upon a time
OUTPUT:  commands commands commands commands commands commands ...
```

### Reading

- GPU1 crossed **loss 30**.
- Unified checkpoint still produces **word-level tokens** in repetition loops (not char noise).
- New lexical items vs earlier probe (e.g. `Fate`, `commands`) alongside recurring ones (`Colorado`, `Population`).
- Not coherent sentences yet. Pipeline + merge path confirmed again.
- Next probes planned near loss ~25 and ~20.


## Operational policy

- Do not stop the 4-GPU digestion while loss is still monotonically improving.
- Re-merge / re-probe generation at later loss thresholds.
- Throughput stable ~1000–1080 tok/s after 30h+ → no evidence of memory leak in this run.

---

*Fractus CTE — continuous thought, phase-routed experts, progressive growth.*
*Training log only. Not a claim of finished language competence.*