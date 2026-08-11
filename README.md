# Fractus CTE

**A living AI that thinks continuously, remembers forever, and grows on its own.**

---

## What is Fractus?

Fractus is not a chatbot. It's not GPT. It's not a transformer.

Fractus is a **Continuous Cognitive Agent** — an AI that works like a brain, not a calculator. Instead of processing input → output in one pass, Fractus **ticks** like a biological system: it maintains a persistent thought state, advances it through multiple blocks of processing, remembers everything across sessions, and can grow new capacity by itself.

### What makes it different from GPT/Claude?

| | GPT-4 / Claude | Fractus |
|---|---|---|
| **Thinking** | One pass, done | Continuous ticks (like a heartbeat) |
| **Memory** | Forgets when context window fills | Remembers forever (survives restarts) |
| **Learning** | Retrain from scratch ($$$) | Learns from every interaction |
| **Growth** | Fixed size forever | Grows new experts at runtime |
| **Mental states** | One mode always | Shifts between cognitive modes |
| **Where it runs** | Corporate cloud | Your machine |

---

## The 12 Building Blocks

| Block | What it does |
|---|---|
| **Continuous Thought Engine** | The brain — thinks tick by tick through 16 blocks |
| **Persistent Memory** | Remembers you across sessions, never forgets |
| **Cognitive Modes** | Shifts mental states (focused, creative, exploratory...) |
| **RAG Knowledge Base** | Learns facts instantly — no retraining needed |
| **Cognitive Plugins** | Hot-swappable modes: analyst, coder, creative, teacher |
| **MetaCognition** | Decides its own actions: retrieve, learn, generate |
| **Progressive Growth** | Grows from 6M to 1B+ params, palier by palier |
| **Self-Modification** | Adds new experts at runtime when it needs them |
| **PhaseRoutedMoE** | Sparse experts routed by oscillator phases |
| **Kuramoto Clock** | A dynamical system that drives routing decisions |
| **Online Trainer** | Learns continuously, one chunk at a time |
| **HF Space** | Live chat demo with shared memory |

---

## How to Use

### Install

```bash
git clone https://github.com/AFKmoney/fractus-cte.git
cd fractus-cte
pip install torch numpy tokenizers matplotlib fastapi uvicorn pydantic
```

### Run tests

```bash
pytest tests/ -q
# → 28 passed (CTE + memory + MoE + multi-block + continuous thought)
```

### Build a corpus

```bash
python scripts/build_quality_corpus.py
```

### Train on CPU (progressive growth)

```bash
# Paliers 0-3: grows from 6M to 350M params
python scripts/train_progressive.py --paliers 0,1,2,3 --accumulation-steps 8
```

### Train on GPU (1B scale)

```bash
# Grows from palier 3 checkpoint to 1B, then trains
python scripts/train_1b_gpu.py \
    --checkpoint checkpoints/fractus_palier3.pt \
    --tokens 500000000 \
    --batch-size 8 \
    --bf16 \
    --accumulation-steps 4
```

### Use the agent

```python
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.memory import PersistentMemory
from fractus.tokenizer import FractusTokenizer

# Build the brain
engine = ContinuousThoughtEngine(
    vocab_size=50257, d_model=128, n_heads=2, d_head=64,
    n_layers=2, n_levels=2, n_oscillators=8, coupling_rank=4,
    n_experts=4, top_k=2, expert_d_ff=128, siren_rank=32)

# Give it memory
memory = PersistentMemory(d_model=128, path="~/.fractus/memory.pt")
engine.attach_memory(memory)

# Think
engine.reset_thought(batch_size=1)
logits, confidence = engine.tick(torch.tensor([42]))
print(f"Confidence: {confidence.item():.2f}")
```

---

## The Growth Path

Fractus grows like a brain — small at first, bigger over time:

| Stage | Size | Blocks | Experts | What it can do |
|---|---|---|---|---|
| Palier 0 | 6.6M | 1 | 4 | Learn basic patterns |
| Palier 1 | 25M | 2 | 8 | Simple text generation |
| Palier 2 | 120M | 4 | 16 | Coherent fragments |
| Palier 3 | 350M | 8 | 32 | Decent text quality |
| **Palier 4** | **1B** | **16** | **128** | **Full language model** |

Each stage inherits the previous one's knowledge. The model never starts from zero.

---

## Architecture (for developers)

```
fractus-cte/
├── fractus/
│   ├── continuous_engine.py      ← The brain (CTE + CTEBlock)
│   │   ├── CTEBlock              One block: attention + Kuramoto + MoE
│   │   └── ContinuousThoughtEngine  Stacks N blocks, carries thought state
│   ├── memory.py                 ← Cross-session persistent memory
│   ├── cognitive_modes.py        ← Unsupervised mental state detection
│   ├── grow.py                   ← Progressive growth operator
│   ├── rag.py                    ← Knowledge base + plugins + metacognition
│   ├── tokenizer.py              ← GPT-2 BPE tokenizer
│   ├── nn/
│   │   ├── moe.py                ← PhaseRoutedMoE (sparse, low-rank, differentiable)
│   │   ├── attention.py          ← Multi-level causal linear attention
│   │   ├── phase_ode.py          ← Kuramoto RK4 oscillators
│   │   └── lazy_siren.py         ← Low-rank weight storage
│   └── train/
│       └── online.py             ← Online trainer (SGD/AdamW, accumulation)
├── tests/                        28 tests
├── scripts/                      Training + corpus + GPU scripts
├── space/                        HF Space demo
├── docs/                         Optimization analysis
├── Fractus_White_Paper.pdf       Technical white paper v2.0
└── arxiv/                        LaTeX source for arXiv submission
```

### Key concepts

**Tick**: one step of thinking. The engine processes an observation, updates its thought state through all blocks, and optionally emits output.

**Thought state**: a vector `h ∈ R^d_model` that persists across ticks. It's the engine's "consciousness" — it carries context forward.

**Chunk**: 32 tokens processed in one forward pass (for speed). The thought state and per-block attention state carry between chunks.

**Expert**: a small neural network (low-rank `W = scale·U@V^T`) that specializes in certain types of thoughts. Only 2 out of 128 are active per token (sparse routing).

**Kuramoto**: coupled oscillators that produce phase vectors. These phases route tokens to the right experts. Think of it as the engine's "internal clock" — different phase patterns = different cognitive modes.

---

## Research Results (Honest)

We tested alternative training methods. Both failed:

- **Expert Decoupled Training (EDT)**: claimed 189x speedup. Reality: 19% worse than standard training. The pre-training objective doesn't align with the final task.
- **Forward-Forward (Hinton 2022)**: local goodness signal. Reality: the model got worse. Local learning can't replace global backpropagation.

**What works**: standard gradient descent + our architectural optimizations = **1345 tokens/second on CPU** (was 4 tok/s before).

---

## License

MIT. Fractus belongs to you, not to a corporation.

## Author

**Philippe-Antoine Robert** — 2026 — rpa.tu@proton.me

## Links

- **GitHub:** [github.com/AFKmoney/fractus-cte](https://github.com/AFKmoney/fractus-cte)
- **HuggingFace:** [huggingface.co/thefinalboss/fractus-cte](https://huggingface.co/thefinalboss/fractus-cte)
- **White Paper:** [Fractus_White_Paper.pdf](Fractus_White_Paper.pdf)
- **arXiv source:** [arxiv/main.tex](arxiv/main.tex)
