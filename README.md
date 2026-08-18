---
license: mit
language:
  - en
  - fr
tags:
  - continuous-thought-engine
  - cognitive-agent
  - mixture-of-experts
  - kuramoto
  - self-modifying
  - progressive-growth
  - decentralized-ai
  - personal-ai
  - neuroscience
library_name: pytorch
pipeline_tag: text-generation
models:
  - thefinalboss/fractus-cte
datasets:
  - thefinalboss/fractus-datasets
---

# Fractus CTE

**A living AI that thinks continuously, remembers forever, and grows on its own.**

**Fractus is NOT a transformer.** It's a Continuous Cognitive Agent — a dynamical system that maintains a persistent thought state, advances it tick by tick through 16 blocks, and routes via Kuramoto oscillator phases. The checkpoint is a living seed: it never freezes, grows at runtime, and trains forever.

> **Status (2026-08-17):** Training live on 8× RTX 5090 — Phase 2, ~3.44B-token corpus, scheduled sampling engaged. Hourly checkpoints synced to this repo. Generation is not yet coherent English (word-salad stage); teacher-forced loss is falling steadily. This is a training-in-progress artifact, not a finished assistant. See [Live Training](#live-training-status).

## Quick Start

```bash
git clone https://github.com/AFKmoney/fractus-cte.git
cd fractus-cte && pip install torch numpy tokenizers

# Load the trained 1B and generate
python -c "
from fractus.continuous_engine import ContinuousThoughtEngine
engine = ContinuousThoughtEngine.from_pretrained('checkpoints/fractus_1b_gpu3.pt')
import torch
logits, confidence = engine.tick(torch.tensor([42]))
print(f'Fractus is thinking. Confidence: {confidence.item():.2f}')
"
```

The `.pt` checkpoint contains the full model (weights + dynamic state). You need this repo's code to run it — Fractus is a custom architecture, not a transformer. Checkpoints are on [HF](https://huggingface.co/thefinalboss/fractus-cte).

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.13-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange)
![Params](https://img.shields.io/badge/params-1.05B-red)
![Active](https://img.shields.io/badge/active%20params-119M-yellow)
![Status](https://img.shields.io/badge/status-training%20live-brightgreen)
![Datasets](https://img.shields.io/badge/datasets-4.2B%20tokens-purple)

---

## What is Fractus?

Fractus is not a chatbot. It's not GPT. It's not a transformer.

Fractus is a **Continuous Cognitive Agent** — an AI that works like a brain, not a calculator. Instead of processing input → output in one pass, Fractus **ticks** like a biological system: it maintains a persistent thought state, advances it through multiple blocks of processing, remembers everything across sessions, and can grow new capacity by itself.

### What makes it different from GPT/Claude?

| | GPT-4 / Claude | Fractus |
|---|---|---|
| **Thinking** | One pass, done | Continuous ticks (like a heartbeat) |
| **Memory** | Forgets when context window fills | Remembers forever (survives restarts) |
| **Learning** | Retrain from scratch ($$$) | Learns from every interaction |
| **Growth** | Fixed size forever | Grows new experts at runtime |
| **Mental states** | One mode always | Shifts between cognitive modes |
| **Where it runs** | Corporate cloud | Your machine |
| **Training** | Fixed, done once | Perpetual, never stops |

---

## Architecture (1.05B total / ~119M active per token)

```
ContinuousThoughtEngine
├── d_model=1280, 16 layers, 16 heads
├── FractalLinearAttention   multi-level causal linear attention (O(L)),
│                            carry state (S, z) persists across chunks
├── Kuramoto phase clock     RK4-integrated oscillators → phase vectors
│                            (learned end-to-end; feeds routing)
├── PhaseRoutedMoE           128 experts, top-2 active per token,
│                            von Mises gate over phases, Farey-sequence
│                            expert phases, load-balance loss in objective
├── Low-rank experts         W = scale · U@Vᵀ (rank 64) → 64× less compute
├── Tied embedding/head      GPT-2 BPE vocab (50257)
└── Persistent thought state residual stream carried across ticks
```

**Parameter accounting:** ~1.05B total, but only ~118.8M are active per token (dense attention + top-2 of 128 sparse experts) — ~183.1M including the tied embedding. An 11.3% sparsity ratio. This is the basis of the adapted scaling target below.

### The 12 Building Blocks

| Block | What it does |
|---|---|
| **Continuous Thought Engine** | The brain — thinks tick by tick through 16 blocks |
| **Persistent Memory** | Vector bank surviving sessions, cosine recall, 5% blend injection, salience head gates impact |
| **Cognitive Modes** | Mental states discovered unsupervised (k-means on Kuramoto phase features): focused, creative, exploratory |
| **RAG Knowledge Base** | Learns facts instantly — no retraining needed |
| **Cognitive Plugins** | Hot-swappable modes: analyst, coder, creative, teacher |
| **MetaCognition** | Decides its own actions: retrieve, learn, generate |
| **Progressive Growth** | Grows from 6M to 1B+ params, palier by palier (`maybe_grow`: width + depth + experts) |
| **Self-Modification** | Adds new experts at runtime when routing is imbalanced (zero-init, placed near the dominant expert) |
| **PhaseRoutedMoE** | Sparse experts routed by oscillator phases |
| **Kuramoto Clock** | A dynamical system that drives routing decisions |
| **Online Trainer** | Learns continuously, one chunk at a time |
| **Vision (`tick_vec`)** | Multimodal input path — image patches drive the engine directly, bypassing the token embedding (CIFAR "eyes" prototype trained while the text run continued) |

---

## Live Training Status (August 2026)

**Hardware:** 8× RTX 5090 — one independent Python process (one "brain") per GPU, no gradient sync; consolidation via mean-merge.

**Phase 2 (current):**
- Corpus: **3,439,171,703 tokens** — 8 int32 numpy memory-mapped shards (~430M/GPU)
- Phase 1 (complete): ~11M tokens on a 1.52B-token `.pt` corpus; weights carried over (no random init)
- Config: batch 2, seq 128, LR 7e-4, SGD momentum 0.9, bf16, TF32, torch.compile off (VRAM), torch 2.11.0+cu128
- **Scheduled sampling:** SS_RATE 0.25, SS_PROB 0.2 — ~25% of steps mix ~20% model-generated tokens into inputs to fight exposure bias
- Loss signals tracked: `tf`/`ema_tf` (teacher-forced CE), `ss`/`ema_ss` (scheduled-sampling CE), `lb` (load balance, stable ~14)
- Throughput: ~900–1100 tok/s per GPU; a full phase-2 pass ≈ 4–5 days

**Hourly crash-recovery sync:** a daemon uploads all 8 per-GPU checkpoints every hour, mean-merges them into `FRACTUS_1B_8GPU_MERGED.pt`, republishes it under the recovery alias `FRACTUS_1B_STAGE2_MERGED.pt`, and writes a manifest with exact per-GPU token offsets. Any pod death is recoverable to the hour.

**Honest generation state:** teacher-forced loss keeps falling (ema_tf ≈ 1.3–1.8 on the best GPUs), but free-run generation is still non-linguistic — diverse word-salad with repetition loops, better with window-based decoding (~25–28 unique tokens vs ~10–12 chunk-based). This is the documented exposure-bias gap, being closed by scheduled sampling + aligned decoding. Low TF loss does **not** mean the model can speak yet. See [Trusted Loss](#trusted-loss-reading-the-numbers).

## Vision — the CIFAR "eyes" prototype

Fractus is not text-only. The engine exposes **`tick_vec(obs_vec)`**: a multimodal entry point that accepts a precomputed `(B, d_model)` vector and injects it directly into the residual thought state — **bypassing the token embedding entirely**. Any modality that can be embedded into `d_model` dimensions can drive continuous thought: images, audio, sensor streams.

```
image → PatchEmbed → (B, d_model) patch vectors
                            │
                            ▼
              engine.tick_vec(patch)     ← no tokenizer involved
                            │
              thought state advances through all 16 blocks
              (same Kuramoto routing + MoE as text)
```

**Proof of concept — CIFAR-10 eyes:** a small CTE + PatchEmbed stack (`fractus/nn/vision.py`) was trained on CPU on real CIFAR-10 images (`fractus_eyes_cifar_final.pt`), demonstrating that image patches can drive the continuous-thought loop. Crucially, this ran as a **parallel track**: the 8-GPU text run was never interrupted for eyes work — GPU text digestion and CPU vision learning proceed simultaneously on the same living system.

This mirrors the biological premise: eyes evolved as a peripheral sense feeding a central dynamical brain, not as a core feature of it. The text 1B is the brain; vision is a front-end that plugs into `tick_vec`.

---

### Adapted Chinchilla target

Standard Chinchilla (20 tokens/param) assumes dense transformers where all parameters train on every token. Fractus violates that: only ~119M of 1.05B params are active per token, and the Kuramoto clock trains end-to-end now. The adapted target is **20 × active params ≈ 2.4–3.7B tokens**; with a warm-started checkpoint, the practical target is **~2.5–3B tokens** — which is why the phase-2 corpus is 3.44B. Since Fractus grows continuously (`maybe_grow`), this is an instantaneous cumulative-exposure requirement that grows with the model, not a final-stop condition. Details: [`docs/2026-08-12-fractus-chinchilla.md`](docs/2026-08-12-fractus-chinchilla.md).

---

## The Surgeries (mid-training interventions, no weight wipes)

A defining discovery of this run: **the brain (.pt) and the code are separable.** Bottlenecks were fixed by live surgery — save the checkpoint, patch the code, reload weights, resume at the exact recorded token offset. Multi-day digestion is never thrown away.

| Phase | Intervention | Result |
|---|---|---|
| **A — Initial** | Last-position-only CE, 4 independent GPUs | Loss fell; generation collapsed into single-token loops |
| **B — Routing surgery** | Kuramoto was frozen under `no_grad` (order parameter r ≈ 0.01–0.03); LB loss was detached → ~70% of experts dead. Fixed: gradients enabled (state kept detached for carry), CE + 0.02·lb, gate temp 1.0 → 2.5, omega scale ×4 | lb ≈ 14 live on all GPUs; experts alive |
| **C — Dense CE** | Replaced last-position CE (1 target / 128 tokens) with CE over all positions | Sharp loss drop; token-to-token chaining enforced |
| **D — Decode surgery** | Phase/thought noise, frequency penalties, cycle bans, forced escape tokens | Loop lock broken; still no coherent English |
| **E — Train/gen mismatch** | Training used causal attention + RK4 Kuramoto; generation used a simpler single-tick Euler path. Aligned decode via `generate_aligned.py` | Decode now matches the training path |
| **F — Loss recalibration** | Cumulative-average CE was misleading near 2.0 → batch CE + EMA; LR 1e-3 → 5e-4 | Trustworthy metrics |
| **G — Scheduled sampling** | Two-pass training: TF CE + LB, plus SS steps mixing model samples into inputs | The current phase; ss/ema_ss now tracked |

Full logs: [`docs/MASTER_RUN_LOG.md`](docs/MASTER_RUN_LOG.md), [`docs/DISCOVERY_LOG.md`](docs/DISCOVERY_LOG.md), [`docs/OPERABILITY_MIDTRAIN.md`](docs/OPERABILITY_MIDTRAIN.md).

### Composable checkpoints

Because shapes stay compatible and manifests record token offsets, the following operations are proven on this run:
- **Parallel independent training** — N GPUs on separate shards
- **Mean-merge** — per-GPU checkpoints averaged into one unified model that still generates (424/440 tensors on the 4-GPU merge; stateful buffers reset)
- **Iterative fusion** — train → merge → train cycles; the model absorbs compatible checkpoints and keeps going
- **Exact-token resume** — across pod reboots and driver crashes
- **Compositional growth ≠ structural growth** — weight averaging vs `maybe_grow` paliers

Details: [`docs/COMPOSABILITY_AND_SURGERY.md`](docs/COMPOSABILITY_AND_SURGERY.md).

---

## Trusted Loss (reading the numbers)

Three metrics, three different questions:

| Metric | What it measures | Where |
|---|---|---|
| `ema_tf` | CE with ground-truth history + continuous internal state — is the model digesting data? | live |
| `ema_ss` | CE after scheduled sampling mixes model-generated tokens in — partial free-run robustness | live |
| **AR** | Warm on 32 true tokens, greedily free-run 32 steps, CE vs truth — actual generation quality | offline |

**A single CE number cannot represent both teacher-forced learning and free-run generation.** Live ema_tf ≈ 1.3–1.4 coexists with AR ≈ 2240 (random guessing over the GPT-2 vocab is ~10.82): free-running compounds every error, and teacher forcing always supplies the correct past. Trust `ema_tf`/`ema_ss` for learning progress, AR + text probes for generation progress. The convergence signal is AR falling toward the ss/tf order of magnitude, plus readable output. Details: [`docs/TRUSTED_LOSS.md`](docs/TRUSTED_LOSS.md).

---

## Research Results (Honest)

**Refuted:**
- **EDT** (Expert Decoupled Training) — all 5 variants ~19–20% worse than from-scratch; MSE objective misaligned with CE, router ignores half the experts.
- **Forward-Forward** (Hinton 2022) — NLL rose from 124 to 221; local objectives can't replace global backprop here.

**Validated:**
- **Progressive growth** — warm start converges faster; trained through palier 3 (350M, loss 23.0, ~5 days CPU).
- **Sparse low-rank MoE** — 2/128 experts = 64× less compute.
- **Open-heart operability** — live surgery on a training model works (see above).
- **Routing pathology as a first-class debug target** — expert-hit histograms and the Kuramoto order parameter catch failures that loss curves hide (they caught the frozen clock and the dead experts).

**Training optimizations (measured):** tied head (~1.1×), head-partial training (~2×), sparse gathered low-rank MoE (8× at 16 experts, 64× at 128), detached-state Kuramoto, gradient accumulation (~1.4×), SGD+momentum over AdamW (~1.37×), batching (335 → 1345 tok/s at B=8), bf16 (~2×). Combined CPU: ~336× over naive. Measured on CPU: 707 tok/s single-stream, **1345 tok/s batched**.

---

## Datasets (4.2 Billion Tokens)

Source of truth: [huggingface.co/datasets/thefinalboss/fractus-datasets](https://huggingface.co/datasets/thefinalboss/fractus-datasets).

| Dataset | Tokens | Content |
|---|---|---|
| **neuro-paradigms-1b** | **~1B** | 100 neuroscience → software architecture paradigms (300 chunked files) |
| **neuro-code-math** | **~900M** | Neuro-inspired coding, mathematics, algorithms (incl. 40 applied-neuroscience topics) |
| **cognitive-skills** | **~780M** | Coding, reasoning, speaking, thinking, understanding |
| **fractus-generated-corpus** | **340M** | Bilingual FR/EN generated by Fractus ontology engine |
| **paradigms-full** | **191M** | 140 paradigms (neuroscience, CS, architecture) |
| **gutenberg-esoteric** | **~58M** | 487 public-domain esoteric / masonic / hermetic books |
| **neuro-arch-full** | **86M** | 60 neuroscience paradigms (neuro-software-architecture) |
| **all-github-repos** | **54M+** | 80+ repos (public + private, secret-filtered) |
| **mega-corpus-v3** | **20M** | Literature, philosophy, occult, masonry, science, medicine |
| **wordnet** | **3M** | 117K dictionary synset entries |
| **Total** | **~4.2B** | |

Tokenized streams: Phase 1 = ~1.52B tokens (`.pt` files); Phase 2 = ~3.44B tokens (8 GPT-2 BPE int32 memmap shards). Phases are kept separate to avoid re-ingesting the same ordered stream.

---

## Applied Neuroscience — the theoretical core

Fractus is a neuroscience-grounded architecture: real brain mechanisms are mapped to software/AI patterns, and that mapping is itself training data. Every entry below is present in the dataset — verified by file listing, not just claimed.

### 100 neuroscience → software-architecture paradigms (`neuro_paradigms_1b`, 300 chunked files)

Each paradigm maps a biological mechanism to an engineering pattern (e.g. *adenosine sleep pressure* → cache-stampede recovery; *myelin sheath* → caching; *hippocampal replay* → trajectory consolidation).

<details><summary><b>show all 100 paradigms</b></summary>

```
adenosine_sleep_pressure            amygdala_prefrontal_topdown       anterior_cingulate_conflict_monitor
apoptosis_self_destructing_service  arc_gene_plasticity_marker        astrocyte_tripartite_synapse
axon_initial_segment_trigger        basal_ganglia_loop_arbitration    bdnf_growth_factor_scaling
bergmann_glia_purkinje              binaural_cross_correlation_localization   brainstem_vital_functions
broca_area_api_generator            calcium_transmitter_coupling      camp_second_messenger_amplifier
cerebellar_forward_model            cholinergic_attentional_filter    circadian_gene_expression
climbing_fiber_error_broadcast      cochlear_compressive_nonlinearity cortical_area_specialization
cortical_minicolumn_pipeline        cortico_cortical_pathways         corticotropin_releasing_hormone
cortisol_slow_stress_recovery       critical_period_learning_rate     dendritic_compartmentalization
endocannabinoid_retrograde          enteric_glia_gut_brain            ependymal_cell_barrier
fusiform_face_service_registry      gaba_inhibitory_bus               gap_junction_electrical_sync
ghrelin_hunger_signal               glomerular_convergence_gateway    glutamate_excitatory_bus
glycine_coagonist_modulator         granule_cell_inhibitory_relay     hair_cell_banks_event_clusters
hippocampal_4ec_loop_replay         histamine_wakefulness_keeper      hox_gene_service_specialization
hypercolumn_module_federation       hypercolumn_sharding              hypothalamus_homeostasis
insula_interoception_monitor        ip3_inositol_cascade              k_complex_event_trigger
kcc2_chloride_shift_inhibitor       leptin_satiety_signal             locus_coeruleus_ne_global_signal
melatonin_circadian_scheduler       microglia_active_surveillance     mitral_tufted_cell_dual_path
morphogen_gradient_config           muller_glia_retina_repair         myelin_sheath_caching
neural_crest_migration_deploy       neuropeptide_y_stress_buffer      ng2_glia_pool_renewal
nitric_oxide_gas_signal             node_of_ranvier_bypass            nrem_slow_wave_cleanup
nucleus_accumbens_reward_routing    oligodendrocyte_myelination_dynamic      orexin_stability_keeper
orientation_column_indexing         oscillatory_phase_locking_io      oxytocin_trust_protocol
parahippocampal_place_topology      parallel_fiber_fanout_aggregation pineal_circadian_release
pinwheel_central_layout             pituitary_master_gland            posterior_parietal_integration
prolactin_parental_care             quantal_release_batching          radial_glia_neural_stem
radial_glial_scaffold               raphe_serotonin_rate_limit        rem_paradoxical_processing
replay_consolidation_trajectory     reticular_activating_system       retinotopic_data_layout
satellite_glial_ganglion            schwann_cell_peripheral_repair    sleep_pressure_forced_maintenance
sleep_spindle_memory_transfer       slow_oscillation_sync             subplate_wait_state
suprachiasmatic_clock               synaptic_vesicle_pool            synaptogenesis_service_wiring
tanycyte_metabolic_sensor           temporal_pole_semantic_cache      thalamocortical_loop_api
tonotopic_stream_partitioning       vasopressin_loyalty_aware_routing vta_dopamine_rpe_scheduler
wernicke_area_api_parser
```
</details>

### 40 applied-neuroscience topics (`neuro_code_math/applied_neuroscience/`)

Deep dives on computational neuroscience theories — the science Fractus's design draws from.

<details><summary><b>show all 40 topics</b></summary>

```
active_inference          axonal_computation         basal_ganglia_circuits     bayesian_brain
cerebellar_computation    consolidation              cortical_minicolumns       cross_frequency_coupling
dendritic_computation     dopamine_reward            entorhinal_grid_cells      free_energy_principle
gamma_oscillations        global_workspace_theory    head_direction_cells       hierarchical_processing
higher_order_theories     hippocampal_formation      homeostatic_plasticity     integrated_information_theory
long_term_depression      long_term_potentiation    metaplasticity             neural_coding
neural_decoding           neural_manifolds          neuromodulation            place_cells
population_coding         predictive_coding          predictive_processing      rate_coding
serotonin_modulation      sharp_wave_ripples         sleep_replay               sparse_coding
spike_timing_dependent_plasticity   temporal_coding  thalamic_reticular_nucleus theta_oscillations
```
</details>

### Foundational researchers & concepts honored in the corpus

**Hebb** (Hebbian learning), **Bi & Poo** (STDP timing curves), **Friston** (free energy / active inference), **Buzsáki** (hippocampal sharp-wave ripples, replay), **Moser & Moser** (grid cells), **Hodgkin & Huxley** (axon dynamics), **Izhikevich** (spike models), **Tononi** (integrated information), **Baars/Dehaene** (global workspace), **O'Keefe** (place cells), **Kandel** (memory consolidation), plus neuromodulators (dopamine RPE, serotonin, oxytocin, vasopressin) and glial biology (astrocytes, microglia, oligodendrocytes, Schwann cells).

---

## How to Use

### Install

```bash
git clone https://github.com/AFKmoney/fractus-cte.git
cd fractus-cte
pip install torch numpy tokenizers matplotlib fastapi uvicorn pydantic
```

### Run tests

```bash
pytest tests/ -q
```

### Train on CPU (progressive growth)

```bash
python scripts/train_progressive.py --paliers 0,1,2,3 --accumulation-steps 8
```

### Train the 1B (GPU, sharded)

```bash
# Build shards, then one process per GPU
python scripts/shard_corpus.py --corpus data/training_corpus.pt --out data/shard_gpu
bash scripts/launch_4gpu.sh        # pattern scales to 8
```

### Use the agent

```python
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.memory import PersistentMemory
from fractus.tokenizer import FractusTokenizer

# Build the brain
engine = ContinuousThoughtEngine(
    vocab_size=50257, d_model=128, n_heads=2, d_head=64,
    n_layers=2, n_levels=2, n_oscillators=8, coupling_rank=4,
    n_experts=4, top_k=2, expert_d_ff=128, siren_rank=32)

# Give it memory
memory = PersistentMemory(d_model=128, path="~/.fractus/memory.pt")
engine.attach_memory(memory)

# Think
engine.reset_thought(batch_size=1)
logits, confidence = engine.tick(torch.tensor([42]))
print(f"Confidence: {confidence.item():.2f}")

# Vision: drive thought with an image patch vector (no tokenizer)
patch_vec = torch.randn(1, 128)          # any (B, d_model) embedding
logits, confidence = engine.tick_vec(patch_vec)
```

---

## The Growth Path

| Stage | Size | Blocks | Experts | What it can do |
|---|---|---|---|---|
| Palier 0 | 6.6M | 1 | 4 | Learn basic patterns |
| Palier 1 | 25M | 2 | 8 | Simple text generation |
| Palier 2 | 120M | 4 | 16 | Coherent fragments |
| Palier 3 | 350M | 8 | 32 | Decent text quality |
| **Palier 4** | **1B** | **16** | **128** | **Full language model (training now)** |

Each stage inherits the previous one's knowledge via zero-padded warm starts. The model never starts from zero.

---

## Repository Layout

```
fractus-cte/
├── fractus/
│   ├── continuous_engine.py      ← The brain (CTE + CTEBlock)
│   ├── memory.py                 ← Cross-session persistent memory
│   ├── cognitive_modes.py        ← Unsupervised mental state detection
│   ├── grow.py                   ← Progressive growth (width + depth + experts)
│   ├── rag.py                    ← Knowledge base + plugins + metacognition
│   ├── tokenizer.py              ← GPT-2 BPE tokenizer
│   ├── nn/
│   │   ├── moe.py                ← PhaseRoutedMoE (sparse, low-rank)
│   │   ├── attention.py          ← Multi-level causal linear attention + carry
│   │   ├── phase_ode.py          ← Kuramoto RK4 oscillators
│   │   └── lazy_siren.py         ← Low-rank weight storage
│   └── train/online.py           ← Online trainer
├── tests/
├── scripts/                      fast_tokenize, shard_corpus, launch_4gpu,
│                                 fast4gpu_boost, generate_aligned, hourly_hf_sync
├── checkpoints/                  per-GPU + merged + recovery alias
├── docs/                         run logs, surgeries, trusted loss, scaling
├── space/                        HF Space demo
├── Fractus_White_Paper_v2.md     White paper v2.0
└── arxiv/                        LaTeX source for arXiv submission
```

---

## Key Concepts

**Tick**: one step of thinking. The engine processes an observation, updates its thought state through all blocks, and optionally emits output.

**Thought state**: a vector that persists across ticks — the engine's "consciousness."

**Chunk**: 32 tokens processed in one forward pass. The thought state and per-block attention state carry between chunks.

**Expert**: a small low-rank network (`W = scale·U@Vᵀ`) that specializes in certain thoughts. Only 2 of 128 active per token.

**Kuramoto clock**: coupled oscillators producing phase vectors that route tokens to experts — learned end-to-end since the routing surgery.

**`tick_vec`**: the multimodal tick — feed any precomputed `(B, d_model)` vector (image patches, embeddings from any encoder) straight into the thought state without tokens. This is how vision plugs in.

**Surgery**: patching code around a preserved checkpoint mid-training, resuming at the exact token offset. Weights are never wiped for a routing, objective, or decode bug.

---

## Limitations (stated plainly)

- Generation is not yet coherent English — word-level repetition loops / lexical noise. Exposure bias is being addressed by scheduled sampling; AR is the metric to watch.
- GPT-2 vocab dominates parameters (81% at d=768) — vocab reduction is a known lever.
- "Remembers forever" and "grows on its own" describe the architecture's design; no independent benchmarks are provided.
- This is a research artifact and a live training run, not a production assistant.

---

## License

MIT. Fractus belongs to you, not to a corporation.

## Author

**Philippe-Antoine Robert** — 2026 — rpa.tu@proton.me

## Links

- **GitHub:** [github.com/AFKmoney/fractus-cte](https://github.com/AFKmoney/fractus-cte)
- **HuggingFace Model:** [huggingface.co/thefinalboss/fractus-cte](https://huggingface.co/thefinalboss/fractus-cte)
- **HuggingFace Datasets:** [huggingface.co/datasets/thefinalboss/fractus-datasets](https://huggingface.co/datasets/thefinalboss/fractus-datasets)
- **White Paper:** [Fractus_White_Paper_v2.md](Fractus_White_Paper_v2.md) / [PDF](Fractus_White_Paper.pdf)
- **Run logs:** [MASTER_RUN_LOG](docs/MASTER_RUN_LOG.md) · [TRAINING_LOG_1B](docs/TRAINING_LOG_1B.md) · [DISCOVERY_LOG](docs/DISCOVERY_LOG.md)
- **arXiv source:** [arxiv/main.tex](arxiv/main.tex)
