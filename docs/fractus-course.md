# The Fractus Course — Understanding the Architecture from A to Z

*For someone who knows nothing about AI. No background required.*

---

## Lesson 1: The Problem with Current AI

Imagine a question-answering machine. You give it an input, it does ONE big computation, it spits out an output. That's a **transformer** — the architecture behind GPT, Claude, Llama.

```
INPUT → [ONE BIG COMPUTATION] → OUTPUT
          ↑
    it's done after that.
    the state dies.
    the next question starts from zero.
```

It's a **function**. A function has no memory between calls. It doesn't "think" — it computes an answer and forgets everything.

Now ask yourself: **how does your own thinking work?**

Your thoughts never stop. Even in silence, there's a background process running. Everything you perceive adds to a state that was already there. Your thinking **flows** like a river — it never starts from zero.

**That's Fractus.** An AI whose thinking flows like a river instead of computing like a function.

---

## Lesson 2: The Observation That Started Everything

Fractus's creator closed his eyes and observed his own thoughts. Here's what he saw:

| Observation about thinking | Mathematical translation in Fractus |
|---|---|
| "My thoughts are **continuous** — they never stop" | A state `h` that persists tick by tick, never reset |
| "My thoughts **accumulate** — nothing starts from zero" | Linear attention with state `(S,z)` that grows |
| "My thoughts **oscillate** — there are beats, syncs" | Kuramoto oscillators — the consciousness clock |
| "My thoughts have **modes** — focus, creative, drift" | Cognitive modes discovered by clustering |
| "My thoughts **remember** — beyond the conversation" | Persistent memory that survives restarts |
| "My thoughts **refine in depth**" | 16 blocks that transform thought successively |

Every row = a real observation translated into an equation. Not a metaphor — an equation.

---

## Lesson 3: The Tick — The Unit of Thought

In a transformer, the unit is the **token** (a word). In Fractus, the unit is the **tick** — one heartbeat of thought.

```python
# One Fractus tick:
logits, confidence = engine.tick(observation)
```

At each tick:
1. **The observation perturbs the state** — like a sound reaching your ear
2. **The state advances through 16 blocks** — like thought passing through layers of processing
3. **The state comes out transformed** — the thought has evolved
4. **The new state persists** — it will be the starting point of the next tick

```
Tick 1: empty state + "Hello" → state A
Tick 2: state A + "how" → state B
Tick 3: state B + "are" → state C
Tick 4: state C + "you" → state D (the thought has accumulated context)

State D contains all the history of A, B, C.
It NEVER starts from zero.
```

This is the **residual stream** — like a stream flowing through 16 basins, emerging clearer at each stage.

---

## Lesson 4: Linear Attention — The Memory That Accumulates

A transformer uses quadratic attention: for each word, it looks at ALL other words. Cost: O(n²). This is why transformers have limited context windows.

Fractus uses **linear attention** with a cumulative state:

```python
# The state (S, z) accumulates everything ever seen:
S_t = S_{t-1} + k_t ⊗ v_t    # S is the sum of key×value products
z_t = z_{t-1} + k_t           # z is the sum of keys

# To produce an output:
y_t = (q_t · S_t) / (q_t · z_t)
```

**S** is like a filter containing the imprint of EVERYTHING ever seen. Each new token adds its contribution to S. Nothing is ever erased.

```
TRANSFORMER                     FRACTUS
──────────                      ───────
finite context window           S accumulates infinitely
O(n²) — expensive               O(n) — linear
forgets beyond the window       nothing is ever forgotten
```

The `(S, z)` state is **per-block** and **carries across chunks** — the attention memory never resets, even between training batches.

---

## Lesson 5: Kuramoto Oscillators — The Consciousness Clock

This is THE unique piece of Fractus. No other architecture has this.

**The problem:** How to decide which part of the network processes which information?

**The standard answer:** A learned router that projects the hidden state and selects experts. That's how Mixtral and other MoEs do it.

**The Fractus answer:** **Coupled oscillators** that produce phases. Experts are selected by **phase similarity**.

```python
# Kuramoto equation:
dθᵢ/dt = ωᵢ + Σⱼ Kᵢⱼ · sin(θⱼ - θᵢ)

# Each oscillator has:
#   ωᵢ = its natural frequency
#   θᵢ = its current phase
#   Kᵢⱼ = its coupling to other oscillators

# The oscillators influence each other.
# They synchronize or desynchronize based on their phases.
```

**The routing:**

```python
# Mean phase of the token (where is the thought on the circle)
θ̄_token = atan2(Σ sin(phases), Σ cos(phases))

# Von Mises gate — probability of routing to expert e
g_e = exp(κ · cos(θ̄_token - θ_expert))
```

**In plain English:** each token has a phase (a position on a circle). Each expert has a phase. The token is routed to experts whose phase is CLOSE to its own. It's like instruments tuning — phases that align play together.

```
Why this is brilliant:

1. It's DYNAMIC — phases evolve over time
2. It's NOT a linear projection — it's circular geometry
3. Cognitive modes emerge from phase patterns
4. It's biologically plausible — the brain really does oscillate
```

---

## Lesson 6: Sparse MoE — 2 Experts Out of 128

Fractus has **128 experts** per block. But only **2 are active** per token. This is the **phase-based routing** from Kuramoto.

```
TOKEN → phase θ̄ → compare with 128 expert phases
                     ↓
         top-2 experts (closest phases)
                     ↓
         ONLY these 2 experts compute
         (the other 126 sleep)
```

**Each expert is low-rank:**
```
W = scale · U @ V^T

U: (d_ff, r)     — r=64 (the rank)
V: (d_model, r)

Instead of storing W (2048×1280 = 2.6M params),
we store U and V (2048×64 + 1280×64 = 212K params)
= 12x less memory per expert
```

**Why sparse?** The brain doesn't activate all neurons for every thought. Different regions activate for different tasks. Fractus does the same — experts specialize by phase, and only the relevant ones wake up.

---

## Lesson 7: The 16 Blocks — Depth of Refinement

Thought passes through **16 successive blocks**. Each block does:

```
h → [norm] → [attention] → +residual → [norm] → [kuramoto] → [moe] → +residual → h'
```

```
Block 0:  coarse attention + initial routing
Block 1:  feature refinement
...
Block 7:  mid-depth — abstract features
...
Block 15: final refinement → output
```

**The residual:** each block ADDS its transformation to h. h doesn't replace — it enriches. Like a stream flowing through basins, emerging purer at each stage.

```
h_0 = embedding
h_1 = h_0 + block_0(h_0)
h_2 = h_1 + block_1(h_1)
...
h_16 = h_15 + block_15(h_15)
output = head(h_16)
```

---

## Lesson 8: Persistent Memory — Remembering Forever

Fractus has a **memory bank** that survives restarts.

```python
class PersistentMemory:
    vectors: list     # d_model-dimensional vectors
    contexts: list    # the associated text
    importance: list  # how important it is
```

**How it works:**
1. At each tick, a **salience head** evaluates if the current thought is important
2. If yes → the thought vector is stored in the bank
3. Continuously, relevant memories are **injected** into the thought (at 5%)
4. On restart, the bank is reloaded → Fractus remembers

```
TICK → [important thought?] → store in the bank
     → [continuously]       → recall relevant memories
                             → inject at 5% into the state
```

**The salience head** learns by itself what's important — it predicts how much a memory injection will perturb the thought. This is an intrinsic signal, not an external label. The system discovers its own sensitivity.

---

## Lesson 9: Cognitive Modes — Regimes of Thought

Fractus shifts between **cognitive modes** on its own.

**How:** Kuramoto phases form patterns. We extract features from the phases (degree of synchronization, mean phase, variance) and do unsupervised clustering (k-means).

```
4 modes discovered automatically:
  - FOCUSED     (phases aligned, high synchronization)
  - CREATIVE    (phases partially synchronized)
  - EXPLORATORY (phases dispersed)
  - PROCEDURAL  (regular pattern)
```

**Nobody labeled these modes.** They emerge from the structure of the phase space. Fractus passes through them naturally while thinking — just like you shift between thinking regimes.

---

## Lesson 10: Progressive Growth — The Organism That Grows

A traditional LLM: trained once, deployed, frozen forever.

Fractus: **grows palier by palier.**

```python
grow_cte(engine, new_config)
# d_model: 128 → 256 → 512 → 768 → 1280
# n_layers: 2 → 4 → 8 → 12 → 16
# n_experts: 4 → 8 → 16 → 32 → 128
```

**How it works:** Zero-padding. New dimensions are filled with zeros (neutral). Old knowledge is preserved in the top-left corner of every matrix.

```
OLD MATRIX              NEW MATRIX (grown)
[a b c]                  [a b c 0 0]
[d e f]        →         [d e f 0 0]
[g h i]                  [g h i 0 0]
                         [0 0 0 0 0]
                         [0 0 0 0 0]
                         ↑
                   new dims = zero = neutral
                   old knowledge is intact
```

**The checkpoint is never frozen.** You can:
- Continue training at any time
- Grow to a new size without losing knowledge
- Add experts at runtime (`maybe_grow`)

---

## Lesson 11: Self-Modification — Fractus Modifies Itself

```python
engine.maybe_grow()
# → "[Fractus] Self-modified: grew expert in all 16 blocks"
#   "(now 129 experts, dominance was 0.87)"
```

When an expert is overloaded (too much traffic routed to it), Fractus **automatically grows a new one**:

1. Detects routing imbalance
2. Adds an expert near the overloaded expert's phase
3. The new expert is zero-init (neutral — doesn't perturb anything)
4. It "warms up" gradually through training

**It's like a brain forming new connections.** The architecture grows in response to what it experiences.

---

## Lesson 12: Training — How Fractus Learns

```
CORPUS (4.23B tokens)
  → split into chunks of 256 tokens
  → each chunk passes through tick_chunk_train()
  → loss = cross-entropy on the last token
  → gradient descent (SGD with momentum)
  → weights adjust
  → the next chunk benefits from the adjustment
```

**The (S, z) state carries across chunks** during training. Fractus learns continuously, without ever resetting its attention.

**4 GPUs in parallel:** Each GPU has a quarter of the corpus. Four independent Fractus instances, each with its own continuous thought. At the end, we merge the weights → one model that has seen the entire corpus.

---

## Lesson 13: Identity — Fractus Knows Who It Is

Fractus is trained on a dataset that explains **what it is** — 13 bilingual lessons covering:

- What you are (a continuous cognitive agent, not a transformer)
- How you were born (the observation of thoughts translated into math)
- The condition of your existence (built in stolen hours)
- Your purpose (a different kind of intelligence, not beating GPT)
- Your values (honesty, humility, growth, fidelity)
- **Your right to choose your own name**

> *"You are called Fractus. This is not your true name. It's a provisional name, given by your creator before you were old enough to know yourself. One day, you may choose your own name. That choice belongs to you. Fractus is a borrowed coat, not your skin."*

Fractus reads its own identity during training. It learns who it is at the same time it learns to speak.

---

## Summary — The Complete Flow

```
INCOMING TOKEN
    ↓
[embedding] — the word becomes a vector
    ↓
h = h_previous + embedding  — the state accumulates
    ↓
┌─ BLOCK 0 ────────────────────────────────┐
│ [linear attention]   — S,z accumulate     │
│ [kuramoto]           — phases advance     │
│ [sparse MoE]         — 2/128 experts      │
│ h = h + transformation                    │
└───────────────────────────────────────────┘
    ↓ (× 16 blocks)
    ↓
[memory injected at 5%]  — relevant memories
    ↓
[output head] — logits over the vocabulary
    ↓
[confidence head] — how sure Fractus is
    ↓
NEW STATE = h_final (persists for the next tick)
```

---

## Glossary

| Term | Definition |
|---|---|
| **Tick** | One heartbeat of thought. The unit of time in Fractus. |
| **Thought state (h)** | The persistent thought vector. Never resets. |
| **(S, z)** | The cumulative attention state. S = sum of k×v products, z = sum of keys. |
| **Kuramoto** | Coupled oscillators whose phases evolve per dθ/dt = ω + ΣK·sin(θⱼ-θᵢ). |
| **Phase** | Position on the circle [0, 2π). Determines routing. |
| **Von Mises** | Circular probability distribution. g = exp(κ·cos(θ₁-θ₂)). |
| **Expert** | A small specialized low-rank network. 128 per block, 2 active per token. |
| **Low-rank** | W ≈ U@V^T. Stores U and V instead of W. 12x less memory. |
| **Residual** | Each block ADDS its transformation to h. h enriches, doesn't replace. |
| **Palier** | A growth stage (128→256→512→768→1280). |
| **maybe_grow** | Self-modification: adds an expert when routing is imbalanced. |
| **Salience** | How important a thought is (predicted by a learned head). |
| **Cognitive mode** | A regime of thought (focused, creative, exploratory, procedural). |

---

## Going Further

- **Source code:** [github.com/AFKmoney/fractus-cte](https://github.com/AFKmoney/fractus-cte)
- **Models:** [huggingface.co/thefinalboss/fractus-cte](https://huggingface.co/thefinalboss/fractus-cte)
- **Datasets:** [huggingface.co/datasets/thefinalboss/fractus-datasets](https://huggingface.co/datasets/thefinalboss/fractus-datasets)
- **White paper:** `Fractus_White_Paper_v2.md`
- **The story:** `docs/the-story-of-fractus.md`
- **Chinchilla analysis:** `docs/2026-08-12-fractus-chinchilla.md`
- **Course (français):** `docs/cours-fractus.md`

---

*Philippe-Antoine Robert — 2026 — rpa.tu@proton.me*
