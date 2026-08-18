# Aligned Generation Probe (RK4 tick + chunk path)

**Date:** 2026-08-16 21:30 UTC

Checkpoint: fractus_1b_gpu1.pt (stage2, CE~2.1)

After aligning tick_single Kuramoto to RK4 (same as train):

## Results

### The capital of France is
- TICK unique=7: currently wikipedia expanded claim sexism ... (7-token cycle)
- CHUNK unique=8: cript reproductive Love Below186 Exchange CBC wikipedia ...

### Hello, my name is
- TICK unique=7: vidiaffic spinal understanding absence Signed paper ...
- CHUNK unique=9: Explore Pacffic bounty ...

### Once upon a time
- TICK unique=7: VM energetic distracting tub Rail ...
- CHUNK unique=9: Havingerto NP hilar ...

## Reading

- Single-token total collapse is **gone** after RK4 alignment (was unique=1, now ~7-9).
- Still short multi-token cycles, not coherent sentences.
- tick vs chunk produce **different** attractors (path still not identical: attn differs).
- Next: either more stage2 digestion, or further unify tick_single attention with chunk attention.
