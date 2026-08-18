# Boost + Outage Pointer

**Updated:** 2026-08-17 03:10 UTC

- Recovery recipe: **docs/BOOST_B4_RECOVERY.md**
- Trainer: **scripts/fast4gpu_boost.py**
- Smoke test: **scripts/smoke_load_merged.py**
- Last HF weight upload: `FRACTUS_1B_STAGE2_MERGED.pt` at **2026-08-16 19:10 UTC (12:10 PM PDT)**
- Pod 3496689 died with Docker RwLayer nil; boost B=4 was live (~940 tok/s) with GPU1 ~203M tokens — those final per-GPU boost ckpts were not yet on HF
