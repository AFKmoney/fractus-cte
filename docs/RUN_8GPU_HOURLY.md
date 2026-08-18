# 8-GPU Relaunch + Hourly HF Sync

Updated: 2026-08-17 18:34 UTC

## Context

Previous 4x5090 pod died (Docker RwLayer nil). Resumed on new 8xRTX 5090 from HF recovery weights.

## Resume point

- Weight: checkpoints/FRACTUS_1B_STAGE2_MERGED.pt
- Code: fractus/ + scripts/fast4gpu_boost.py
- Data: 8 shards x 40M tokens from thefinalboss/fractus-datasets
- Torch: 2.11.0+cu128 (required for 5090)

## Training config

- 8 independent GPU processes
- BATCH=2 (B=4 OOM on this stack)
- SEQ=128 LR=7e-4
- SS sequential 0.25/0.20
- compile disabled for VRAM
- all start from STAGE2_MERGED

## Hourly backup

Daemon: /workspace/hourly_hf_sync.py every 3600s

1. Upload fractus_1b_gpu0.pt through gpu7.pt (8 brains)
2. Mean-merge to FRACTUS_1B_8GPU_MERGED.pt
3. Also publish as FRACTUS_1B_STAGE2_MERGED.pt recovery alias
4. HOURLY_MANIFEST.json

Policy: 8 individual + 1 unified every hour.

## Related docs

- COMPOSABILITY_AND_SURGERY.md / OPERABILITY_MIDTRAIN.md
- TRUSTED_LOSS.md / LOSS_VS_GEN.md
- DECODE_SURGERY.md / GENERATE_ALIGNED.md
- BOOST_B4_RECOVERY.md
- MASTER_RUN_LOG.md
