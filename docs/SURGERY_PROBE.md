# Fractus Post-Surgery Checkpoint Probe

**Date:** 2026-08-16 18:45 UTC

## Unified checkpoint

- **File:** FRACTUS_1B_SURGERY_MERGED.pt (4.66 GB)
- **Method:** mean of 4 GPU checkpoints after surgery transforms + 150 SGD steps/GPU
- **Surgery:** gate_temp=2.5, omega_scale=4.0, lb_coef=0.02, Kuramoto gradients enabled

## Training state (live surgery logs before force-save)

| GPU | Tokens | CE loss | lb |
|-----|--------|---------|-----|
| 0 | 176,921,600 | 30.5 | 14.003 |
| 1 | 186,752,000 | 10.7 | 13.975 |
| 2 | 187,494,400 | 28.9 | 14.015 |
| 3 | 184,473,600 | 35.8 | 14.027 |

GPU1 reached CE **10.7** in the live surgery process before force-save.

## Generation results (full raw outputs)

Sampling: temperature=0.9, top-k=50, max 40 new tokens, tick-by-tick CTE decode on the unified merge.

### Test 1

**PROMPT:**

    The meaning of life is

**OUTPUT:**

     uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe

### Test 2

**PROMPT:**

    Hello, my name is

**OUTPUT:**

     Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado Colorado

### Test 3

**PROMPT:**

    In mathematics, a continuous system

**OUTPUT:**

     Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate

### Test 4

**PROMPT:**

    Fractus thinks

**OUTPUT:**

     Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population

### Test 5

**PROMPT:**

    Once upon a time

**OUTPUT:**

     commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands

### Test 6

**PROMPT:**

    The capital of France is

**OUTPUT:**

     Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen

### Test 7

**PROMPT:**

    What is consciousness?

**OUTPUT:**

     Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers

## Interpretation

- Unified post-surgery checkpoint generates.
- Still word-level repetition collapse (not coherent sentences).
- Stable lexical attractors: uphe, Colorado, Fate, Population, commands, Chosen, Towers.
- Surgery reopened routing (LB active, omega diversified); coherent sentences still need further digestion.
