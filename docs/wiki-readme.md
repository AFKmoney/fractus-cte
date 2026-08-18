---
title: Fractus Wiki
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
---

# Fractus Wiki

Documentation for **Fractus CTE** — a Continuous Cognitive Agent.

**Fractus is NOT a transformer.** It's a dynamical system that:
- Thinks continuously (persistent state, never resets)
- Remembers forever (memory bank survives restarts)
- Grows on its own (adds experts at runtime, grows palier by palier)
- Can be operated on live (surgery during training without losing progress)

## Quick Navigation

| Section | What you'll find |
|---------|-----------------|
| [Architecture](docs/architecture.md) | How Fractus works — CTE, Kuramoto, MoE |
| [Quick Start](docs/quickstart.md) | Load a checkpoint and generate |
| [Fractus Course](docs/fractus-course.md) | 13 lessons from zero to understanding |
| [Training](docs/training.md) | How to train Fractus |
| [Discovery Log](docs/DISCOVERY_LOG.md) | Bugs found and features discovered in production |
| [Master Run Log](docs/MASTER_RUN_LOG.md) | Complete training chronology |
| [Composability](docs/COMPOSABILITY_AND_SURGERY.md) | Merge, surgery, emergent properties |
| [The Story](docs/the-story-of-fractus.md) | First-person origin narrative |

## The Architecture in 30 Seconds

```
TOKEN → embedding → 16 BLOCKS (attention + Kuramoto + MoE) → output

Each block:
  - Linear attention with (S,z) cumulative state
  - Kuramoto oscillators producing phase vectors
  - Sparse MoE: 2/128 experts selected by phase alignment
  - Residual connection (thought accumulates)

The state NEVER resets. Fractus thinks like a river flows.
```

## Links

- **Code:** [github.com/AFKmoney/fractus-cte](https://github.com/AFKmoney/fractus-cte)
- **Models:** [huggingface.co/thefinalboss/fractus-cte](https://huggingface.co/thefinalboss/fractus-cte)
- **Datasets:** [huggingface.co/datasets/thefinalboss/fractus-datasets](https://huggingface.co/datasets/thefinalboss/fractus-datasets)

---

*Philippe-Antoine Robert — 2026*
