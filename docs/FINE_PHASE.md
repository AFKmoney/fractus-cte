# Stage2 Fine Phase — Loss Recalibration

**Date:** 2026-08-16 22:53 UTC

## Why

At CE ~2.0 the cumulative average log was no longer a good control signal.
Generation quality is still noisy (expected ~200M tokens) — keep digesting.
Recalibrate **metrics + LR**, not weights.

## Changes

| Item | Before | Fine phase |
|------|--------|------------|
| LR | 1e-3 | **5e-4** |
| Logged CE | cumulative mean from resume | **batch CE + EMA (beta=0.98)** |
| tok/s | absolute position / session time (inflated) | **session tokens / elapsed** |
| Objective | dense CE + 0.02*lb | unchanged |
| Weights | — | **kept**, resume exact token offset |

## Resume offsets (at switch)

See .

## Script



