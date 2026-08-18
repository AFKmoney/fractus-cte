# Stage2 Generation Probe

**Date:** 2026-08-16 19:09 UTC

## Checkpoint

- FRACTUS_1B_STAGE2_MERGED.pt (4.66 GB)
- Mean of 4 GPU stage2 dense-CE checkpoints
- Live CE at merge time (approx): GPU1 ~3.6, GPU2 ~5.4, GPU0 ~16.7, GPU3 ~15.2

## Full generation outputs

### Test 1

**PROMPT:**

    The meaning of life is

**OUTPUT:**

     uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe uphe

### Test 2

**PROMPT:**

    Hello, my name is

**OUTPUT:**

     operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating operating

### Test 3

**PROMPT:**

    In mathematics, a continuous system

**OUTPUT:**

     Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate Fate

### Test 4

**PROMPT:**

    Fractus thinks

**OUTPUT:**

     Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population Population

### Test 5

**PROMPT:**

    Once upon a time

**OUTPUT:**

     commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands commands

### Test 6

**PROMPT:**

    The capital of France is

**OUTPUT:**

     Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen Chosen

### Test 7

**PROMPT:**

    What is consciousness?

**OUTPUT:**

     Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers Towers

## Reading

- Still word-level repetition collapse.
- New attractor vs earlier probe: operating (was Colorado on Hello).
- Dense CE lowered train loss hard; decode fixed-point not broken yet.
- Stage2 continues on 4 GPUs.
