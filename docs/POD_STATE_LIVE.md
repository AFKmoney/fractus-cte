# Pod state live — 2026-08-17T23:50:38.680454+00:00

## Training
- Phase: **phase2** (raw HF dataset tokenized)
- 8x RTX 5090, BATCH=2, SEQ=128, LR=7e-4, SS_RATE=0.25
- Shard: `data/shard_phase2_gpu{0-7}.npy` (~430M tokens each, memmap)
- Total phase2 corpus: **3,439,171,703** tokens
- Weights continuous from phase1 (~11M tokens on 1.52B .pt corpus)

## Anti re-ingest
- Phase1 stream (1.52B pt) separate from phase2 (raw jsonl/gz)
- RESUME_MANIFEST_8GPU.json tracks offsets
- PHASE_SWITCH.json records switch timestamp

## Artifacts on HF
- Model repo `thefinalboss/fractus-cte`: checkpoints, scripts, docs
- Dataset repo `thefinalboss/fractus-datasets`: original + phase2 tokenized npy shards

## Do not destroy pod
