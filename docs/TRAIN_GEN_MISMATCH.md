# Train/Gen Path Mismatch — Root Cause

## Finding

Training uses  → :
- Multi-level causal linear attention over the full chunk
- Kuramoto **RK4** integrate
- MoE over all positions

Default generation used  → :
- Single-step attention (level-0 style path)
- Kuramoto **one Euler step** (theta + 0.1 * derivative)
- MoE on one token

So the model was trained on one dynamics and decoded with another.

## Fixes

1. **Code:**  Kuramoto now uses  (same as train).
2. **Decode:** Prefer  / sliding-window chunk gen for train-aligned logits.
3. **Decode surgery:** still useful for attractor lock; does not replace path alignment.

## Expected

Aligning dynamics removes a major source of collapse/incoherence. Coherent language still depends on further digestion under stage2 dense CE.

