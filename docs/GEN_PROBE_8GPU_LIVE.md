# Live Gen Probe 8-GPU

Updated: 2026-08-17T19:29:56.573368+00:00

- ckpt: 
- loaded tensors: 391

## Prompt


- unique tokens: 5



## Prompt


- unique tokens: 6



## Prompt


- unique tokens: 5



## Prompt


- unique tokens: 4



## Prompt


- unique tokens: 6




## Training snapshot (same time)

| GPU | tokens | ema_tf | tok/s |
|-----|--------|--------|-------|
| 0 | ~8.9M | ~9.1 | ~955 |
| 1 | ~8.7M | ~8.9 | ~933 |
| 2 | ~9.9M | ~8.0 | ~1062 |
| 3 | ~8.7M | ~8.7 | ~932 |
| 4 | ~8.8M | ~9.4 | ~941 |
| 5 | ~9.2M | ~9.2 | ~991 |
| 6 | ~9.0M | ~8.9 | ~960 |
| 7 | ~9.4M | ~8.6 | ~1010 |

## Interpretation

- Teacher-forced loss continues to fall on all 8 GPUs (ema_tf ~8-9 after ~9M tokens/shard).
- Free-run greedy generation still shows **single-token attractor collapse** (repeat last word / commas).
- This matches prior stage findings: knowledge in weights (TF) != free-run coherence yet (AR gap / decode dynamics).
- Unique tokens per prompt ~4-6 out of 48 generated.

## Next

- Keep SS training running.
- Optional: re-apply decode surgery / temperature / n-gram ban at inference.
- Do not read TF loss alone as "model speaks".
