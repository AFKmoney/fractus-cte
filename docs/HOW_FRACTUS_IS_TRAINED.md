# How Fractus Is Trained

**Updated:** 2026-08-18 00:17 UTC  
**Author:** Philippe-Antoine Robert  
**Model repo:** https://huggingface.co/thefinalboss/fractus-cte  
**Dataset:** https://huggingface.co/datasets/thefinalboss/fractus-datasets

This note explains the actual training procedure for Fractus-1B (Continuous Thought Engine): multi-GPU runs, corpus handling, mid-train surgery, and recovery after host failure.

---

## 1. What Fractus is (training-relevant)

Fractus is not a standard decoder-only transformer trained only with next-token CE on a frozen residual stream.

| Component | Role in training |
|-----------|------------------|
| Continuous thought state | Carries state across ticks |
| Kuramoto oscillators | Phase dynamics for temporal structure / routing |
| Phase-routed MoE | Sparse experts selected via phase/gates |
| Dense CE path (tick_chunk_train) | Teacher-forced sequence loss aligned with gen path |
| Scheduled sampling (SS) | Mix of ground-truth and model predictions as inputs |
| Load-balance loss | Keeps experts alive |
| Gate temperature | Controls routing softness |

Weights live in .pt checkpoints. Code lives in fractus/. Both are required to run.

---

## 2. Corpus (source of truth = HF dataset)

**Source of truth:** thefinalboss/fractus-datasets

A local full_corpus.pt (~4.23B) was historically a concatenation artifact built from this dataset. If that file is missing after a host crash, rebuild from the same HF dataset. The dataset is not lost.

### Dataset layout

- neuro_paradigms_1b/ — neuroscience to architecture paradigms (jsonl.gz)
- cognitive_skills/ — skill / coding / reasoning JSONL
- neuro_code_math/ — math + code + applied neuroscience
- data/training_corpus.pt and datasets/*.pt — already-tokenized streams
- literature / esoteric / repos / identity subsets

### Tokenized streams used in practice

| Stream | Approx size | Location |
|--------|-------------|----------|
| Phase-1 tokenized .pt union | ~1.52B tokens | dataset data/ + datasets/*.pt |
| Phase-2 full raw tokenize | 3.44B tokens | tokenized/phase2/shard_phase2_gpu0-7.npy |

Phase-2: stream all relevant JSONL/JSONL.GZ, GPT-2 BPE encode, write 8 equal int32 numpy memmap shards.

Anti re-ingest: phase-1 and phase-2 are separate streams. After phase-1 progress, switch to phase-2 rather than restarting the same ordered stream from token 0 when the goal is new data.

---

## 3. Multi-GPU training layout

Target: 8x RTX 5090 (recovery). Earlier: 4x.

| Setting | Typical value |
|---------|----------------|
| Processes | 1 Python process per GPU |
| CUDA_VISIBLE_DEVICES | equals GPU_ID |
| Batch | 2 or 3 (B=4 can OOM with SS) |
| Sequence length | 128 |
| LR | 7e-4 (SGD momentum 0.9) |
| SS_RATE | 0.25 |
| SS_PROB | 0.2 |
| LB_COEF | 0.02 |
| Gate temperature | 2.5 |
| TF32 + cudnn.benchmark | on |
| torch.compile | often off (VRAM) |
| Large shards | .npy memmap |

Script: scripts/fast4gpu_boost.py

Each GPU reads only its shard and writes checkpoints/fractus_1b_gpu{i}.pt

### Launch example



Repeat for GPUs 1-7. Never launch multiple workers without unique CUDA_VISIBLE_DEVICES.

---

## 4. Loss signals

| Signal | Meaning |
|--------|---------|
| tf / ema_tf | Teacher-forced dense CE |
| ss / ema_ss | Loss under scheduled-sampling inputs |
| lb | Load-balance term |

Do not equate low TF loss with coherent free-run text. TF can be strong while greedy AR still mono-token collapses until SS + decode path close the train/gen gap.

See docs/LOSS_VS_GEN.md, docs/TRUSTED_LOSS.md, docs/GEN_PROBE_*.

---

## 5. Checkpointing and merge

- Per-GPU: fractus_1b_gpu0.pt ... gpu7.pt
- Mean-merge floating tensors across GPUs -> unified brain (e.g. FRACTUS_1B_PHASE2_LIVE_MERGED.pt)
- Hourly HF sync of 8 individuals + merged (Xet for binaries)
- RESUME_MANIFEST_8GPU.json stores per-GPU token offsets

Checkpoints can be merged, reloaded, continued. Mid-train edits are possible when careful.

---

## 6. Mid-train operability

Distinctive vs typical LLM pretrain:

- Probe experts/phases offline without destroying live checkpoint state
- Decode-path surgery (align tick_chunk vs single-step, anti-collapse)
- Merge parallel trained shards into one model
- Continue after host migration from HF weights

See COMPOSABILITY_AND_SURGERY.md, OPERABILITY_MIDTRAIN.md, DECODE_SURGERY.md.

---

## 7. Recovery playbook (host death)

1. Treat HF as source of truth
2. New pod + torch matching GPU arch (5090 needs recent CUDA builds)
3. Code + dataset from HF
4. Restore weights from checkpoints/fractus_1b_gpu*.pt or merged
5. Restore shards from tokenized/phase2/*.npy or rebuild
6. Resume START_TOKEN from RESUME_MANIFEST_8GPU.json
7. Re-enable hourly Xet upload

Phase switch record: PHASE_SWITCH.json

---

## 8. Current production recipe (phase 2)

1. Dataset fully present from HF
2. Phase-2 tokenization done: 3,439,171,703 tokens -> 8x npy shards (~430M/GPU)
3. Train 8-way BATCH=2, SS on, memmap shards
4. Weights continued from phase-1 (not random init)
5. Checkpoints + phase-2 shards + manifests on HF

Throughput ~900-1100 tok/s/GPU at B=2. One full phase-2 pass ~4-5 days wall-clock.

---

## 9. What done is not

Finishing tokens is not a finished model.

Progress criteria:

1. Stable multi-GPU run
2. TF loss trending down without NaNs
3. SS loss not exploding vs TF
4. Gen probes: rising uniqueness / less mono-token lock
5. Checkpoints recoverable purely from HF

---

## 10. One-sentence summary

Fractus is trained as eight parallel continuous-thought engines on sharded token streams from the HF neuroscience-grounded dataset, optimized with dense teacher-forced CE plus scheduled sampling, checkpointed per GPU, mean-merged and uploaded hourly, and designed so training can be paused, surgically modified, merged, and resumed without treating the run as a single disposable monolith.

---

Machine notes from the live 8x5090 recovery run. Update when the recipe changes.
