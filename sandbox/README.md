# Fractus Sandbox — CTE Cockpit

A real-time dashboard to **see every metric** of the Continuous Thought Engine and
**control/correct** it on the fly. Not a chat — a cockpit.

![status](https://img.shields.io/badge/status-live-success)

## What it shows (8 panels)

| Panel | What you see |
|-------|-------------|
| **Live Monitor** | `‖thought‖`, confidence gauge, tick count, lb_loss, salience_loss, memory count, μ/σ of thought |
| **Oscilloscope** | Kuramoto phases per block, animated on the unit circle (one cell per block) |
| **Expert Routing** | Bar chart of `expert_hits` — which experts fire, dominance % |
| **Training Curve** | Loss + perplexion over time (live, last 100 steps) |
| **Memory Bank** | All persistent memories: context, importance, vector norm — delete individually |
| **Cognitive Mode** | Current mode (focused/creative/exploratory/procedural) + probability bars |
| **Chat + Controls** | Talk to Fractus, step a single tick, reset, grow, inject tokens |
| **Config + Params** | Toggle memory, salience bias slider, full param breakdown (d_model, layers, experts, rank) |

## What you can do

- **See** every internal metric update live over WebSocket (push on every tick).
- **Control**: chat, single-tick step-through (debugging), reset thought state, force `maybe_grow`, inject arbitrary tokens.
- **Correct**: delete a bad memory, add a manual one, or retrain on text (gradient correction via `tick_chunk_train`).
- **Debug**: tick-by-tick inspection of the residual stream, routing distribution, and oscillator phases.

## Run

```bash
pip install fastapi "uvicorn[standard]"
cd sandbox
uvicorn app:app --host 0.0.0.0 --port 7860
# open http://localhost:7860
```

## Endpoints

**Monitoring (GET):** `/api/state` · `/api/phases` · `/api/routing` · `/api/mode` · `/api/memory` · `/api/training` · `/api/params`

**Control (POST):** `/api/chat` · `/api/tick` · `/api/reset` · `/api/inject` · `/api/train` · `/api/grow` · `/api/config` · `/api/memory/add` · `/api/memory/delete/{idx}`

**Live:** `WS /ws/live` — pushes full engine state on every tick.

## Config

The sandbox boots a small CTE (d_model=128, 2 blocks, 8 experts, ~6.8M params) so it
runs on CPU in real-time. To drive a bigger/loaded engine, edit `get_engine()` in
`app.py` to load a checkpoint via `grow.py` or `torch.load`.
