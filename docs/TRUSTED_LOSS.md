# Trusted Loss Protocol for Fractus-1B

**Updated:** 2026-08-17 00:57 UTC

This document defines which loss numbers are trustworthy for what.
A single CE number cannot represent both teacher-forced learning and free-run generation.

---

## 1. Three metrics

| Metric | Definition | Trust for |
|--------|------------|-----------|
| **ema_tf** (live) | Dense CE with ground-truth history + continuous thought state during training | Is training digesting? |
| **ema_ss** (live) | CE after mixing model-sampled tokens into inputs (scheduled sampling pass) | Partial free-run robustness |
| **AR** (eval) | After warm start on true tokens, feed model greedy predictions; CE vs ground-truth next tokens | Generation quality trend |

Reference: uniform random over GPT-2 vocab is about log(50257) ≈ **10.82**.

---

## 2. AR evaluation protocol (trusted gen loss)

Script:  (run offline on a checkpoint)

1. Load checkpoint (e.g. fractus_1b_gpu1.pt)
2. Pick evaluation positions near the current train head
3. **Warm:** run tick_chunk on 32 true tokens
4. **Free-run:** for 32 steps:
   - compute CE(logits, true_next_token)
   - take argmax prediction
   - feed prediction back via tick_chunk
5. Report mean CE (AR) and token match rate (ARacc)

Also report cold/short TF on a 128-token chunk for contrast (not the same as live ema_tf).

---

## 3. Measured numbers (GPU1)

### Live training (SS phase, continuous state)

- GPU 0: 
- GPU 1: 
- GPU 2: 
- GPU 3: 

Interpretation: GPU1 **ema_tf ~1.3–1.4** means teacher-forced next-token prediction is strong under the live continuous regime.

### Offline AR eval (trusted gen)

- Checkpoint pos head: 200806400
- TF_mean (cold-ish offline chunks): **39.512**
- **AR_mean (free-run): 2240.174**
- **ARacc_mean: 0.0**

Raw note from eval JSON: AR is free-run greedy CE vs ground truth after 32-token warm start. Trust AR for gen trend.

Additional offline warm continuous TF (12 chunks of B=2, seq=128 on same ckpt): last-4 mean ≈ **16.6**.
This is higher than live ema_tf because offline warm does not reproduce hours of continuous state + online SGD.

---

## 4. What to trust

| Question | Use |
|----------|-----|
| Is the run still learning? | **ema_tf** (and ema_ss) in train logs |
| Is generation getting better? | **AR** + actual text probes |
| Can I quote one number in a paper abstract? | Report **both** ema_tf and AR; never AR-implied quality from tf alone |

### Rule

- **ema_tf ≈ 1.3** does **not** mean the model speaks.
- **AR >> 10 with ARacc ≈ 0** means free-run is still off-manifold.
- Progress toward speech is **AR falling** toward ss/tf order of magnitude, plus readable text probes.

---

## 5. Why live tf can be ~1 while offline AR is huge

1. Teacher forcing always feeds the correct past
2. Live training carries thought/attention/phase state across millions of tokens
3. Free-run compounds prediction errors every step
4. Train/gen path mismatches (now mostly patched) amplified the gap earlier

Scheduled sampling exists specifically to shrink this gap during training.

---

## 6. Related

- docs/LOSS_VS_GEN.md
- docs/STAGE2_SS_GEN_PROBE.md
- docs/MASTER_RUN_LOG.md
- scripts/fast4gpu_stage2_ss.py
