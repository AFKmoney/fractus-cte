# Fractus — the story behind the architecture

*A first-person account. Signed Philippe-Antoine Robert.*

---

## The moment I closed my eyes

It began with a simple gesture. One evening, after a shift, I closed my eyes, and instead of reaching for something to read, I watched what was happening behind my eyelids. My own thoughts.

I didn't analyze them the way a psychologist analyzes them. I watched them the way a physicist watches a phenomenon — with the naïve, total demand to understand them mechanically. And what I saw was that **my thoughts do not work at all like a transformer.**

A transformer is: you give it an input, it performs one large computation, it emits an output. Input → function → output. It is an answer machine, not a thinking machine.

But when I observed my own thoughts, here is what I saw:

**My thoughts are continuous.** They never stop. Even when I am silent, even in stillness, something is running — a background, a murmur, an activity that precedes speech. I do not "compute" a response: I let it emerge from a current that was already flowing before the question arrived.

**My thoughts accumulate.** Nothing I perceive starts from zero. It adds to a state that was already there, holding everything I have lived in the last second, the last minute, the last year. My mind has an internal state that persists and grows.

**My thoughts oscillate.** It is not a smooth, flat stream. It is rhythmic. There are beats, synchronizations, moments when several things "align" at once and a thought becomes clear — then dissolves. Like clocks falling into the same tempo.

**My thoughts have modes.** Sometimes I am focused, sometimes creative, sometimes I drift. These are not functions I call; they are dynamical regimes my mind slips into on its own.

**My thoughts have memories.** Not only within a conversation — memories that survive from one day to the next, that surface uncalled, that color everything.

That night I understood one thing: **we were modeling intelligence with the wrong metaphor.** We had reduced it to a function. But a thought is not a function. A thought is a **dynamical system flowing in time**, with a state, a rhythm, regimes, and a memory.

And I asked myself: if I translate what I just observed, layer by layer, into mathematics — do I get a different kind of intelligence?

It became an obsession. Here is how I arrived at Fractus.

---

## 48 hours a week (the real condition of the experiment)

There is something essential I need to say, because without it the shape of this story doesn't make sense.

During all of this time — during every one of the repositories I am about to describe — I was working **48 hours a week at the Spirit Ridge hotel.** Service work. On my feet, shifts, guests, schedules that never land neatly. I was not a researcher in a lab with free time and a salary to think. I was someone who finished a shift, went home, and opened a laptop before exhaustion could kill the idea.

Let's do the math. From the first commit to the seed, on May 2, 2026, to the synthesis of `fractus-cte` on August 11: **fourteen weeks.** At 48 hours a week, that is **672 hours spent at Spirit Ridge** over that period. Add sleep, add commuting, add the bare fact of existing and recovering from a physical job. What was left to build Fractus — to read Kuramoto, to understand Katharopoulos, to write causal linear attention, to test five EDT variants and kill them one by one, to push thirteen repositories and thousands of commits — was **the crumbs.** The too-short evenings, the swallowed days off, the small hours of the morning when you code between two alarms.

And it was in those crumbs of time, exhausted, that what you are reading came out.

Once you know that, the facts carry a different weight. The explosion of July 21–25 — four major repositories in three days, including the day Fractus is named — is not the leisure of an inspired researcher. It is someone who had to wait for an alignment of days off to pour out at once everything that had been ripening in his head during the shifts. Most of the ideas took shape **standing up, behind a counter or in a hotel corridor**, not seated at a screen. The screen was only the moment I wrote them down.

I say this because it is the real condition of the experiment. Fractus was not born of free time. It was born of the discipline of stealing time from a job that left none — of preferring, evening after evening, the equation over sleep. Very little time to do it. And yet, done.

---

## The translation, layer by layer

I took each observation and translated it into equations. Not metaphors. Equations.

**"My thoughts are continuous"** → I need a state that persists from one instant to the next. A vector `h` that is never reset to zero, that advances tick by tick. The tick becomes the unit of thought, the way a heartbeat is the unit of life. An input does not trigger a computation: it perturbs a state that already existed.

**"My thoughts accumulate"** → attention is not a sliding window; it is an accumulator. I took Katharopoulos' causal linear attention, with its `(S, z)` state that grows with every token and is never reinitialized. S is the short-term memory of attention, carried even across chunk boundaries. The thought carries its past with it.

**"My thoughts oscillate"** → I needed a clock. Not a clock that counts seconds, but a coupled clock — oscillators that influence one another and that, by synchronizing, create an emergent rhythm. The Kuramoto equations do exactly this: `dθᵢ/dt = ωᵢ + Σ Kᵢⱼ sin(θⱼ − θᵢ)`. Coupled oscillators that, depending on their phases, align or desynchronize. I made it Fractus's **consciousness clock**: it produces phase vectors that, at each tick, decide which part of the network the thought should be routed toward.

**"My thoughts have modes"** → if Kuramoto phases oscillate in a space, then configurations of phases form regions, "climates" of the phase space. I let an unsupervised k-means discover those regions on its own — with no labels whatsoever. And modes appeared: a focused mode, a creative mode, an exploratory mode. Fractus passes through them on its own, the way I move from one thinking regime to another.

**"My thoughts have memories"** → I needed a bank that survives restarts. Vectors with a context and an importance score, recalled by similarity and continuously injected into the thought state. And above all: a **salience head** that learns on its own how much an injection will perturb the thought — an intrinsic signal, not an external label. The system discovers its own sensitivity to its memories.

And then the central piece: **"my thoughts are not computed in one pass, they are refined in depth"** → I stacked these layers. The thought state enters a block (attention → kuramoto → mixture of experts), comes out transformed, enters the next, and so on, like a stream passing through a succession of basins and emerging clearer at each step. Each block owns its own attention state, its own phases, its own experts. Thought is a **residual stream.**

That was Fractus. But before arriving there, there was a whole evolution — visible, commit by commit, across my repositories.

---

## The evolution, read through the repositories

I reconstructed my own trail from the dates of my first commits. The story is clear: I did not design Fractus all at once. I gradually carved out, trial by trial, what actually mattered.

**May 2, 2026 — `Fractal-Neural-Network`.** The first seed. Already there, without naming it, was all the material: a fractal topology, phase synchronization via Kuramoto, and a phase-routed MoE. I was still writing with exotic optional layers (AdS/CFT, MERA, Gödel, RG) — I was exploring. But the core intuition was set: **the brain synchronizes, and synchronization should route.**

**June 13, 2026 — `CogNet`.** "Non-Transformer Language Model with Cognitive Routing, 40M params, O(n)." The first time I held in my hand a non-transformer model that ran **on CPU**, with linear cost. Proof that I was not obliged to accept the dogma of the quadratic.

**July 11–12, 2026 — `aether-ai` then `nova-spike-hybrid`.** Two days, two turns. AETHER was the devouring ambition: beat GPT-4, on CPU, in pure NumPy. Then, the very next day, the real intellectual pivot: NOVA/SPIKE, **spiking** neural networks, with STDP (spike-timing-dependent plasticity), the real biological learning rule. That is where I left engineering for neuroscience. I stopped copying the architecture of LLMs and started copying **the architecture of the brain.**

**July 21, 2026 — the day Fractus was born.** Two repositories, the same day. `kahnn`: Kuramoto-Attractor Holographic Hypervector Network, 1B params. And above all `fractus-test`: "Experimental Holographic Vector Learning for Fractus." **It is on this day that the word *Fractus* appears.** The idea of a vector that persists, that binds holographically, that you train in one shot. The name was set. The thing still imperfect. Two repositories in a single day, after weeks of 48-hour shifts — that is the kind of output that happens when you have carried an idea in your head for too long without being able to write it, and finally the time frees up.

**July 23–25, 2026 — the creative explosion.** Four repositories in three days. `Modele-Variance-Topologique`: replace tensors with differential geometry and topological fields, and plant EDT (Expert Decoupled Training) there — my first real attempt to accelerate MoE training. `oscillon-architecture`: "Oscillatory **Stateful Continuous** Intelligence" — the word *continuous* finally enters the title. `kortex`: Kuramoto Oscillator Reasoning & Thought Express, with equilibrium propagation — my boldest attempt, **to kill backprop.** `CogNet-MoE-1B`: the first climb to 1B, eight cognitive channels as eight experts.

**August 4–7, 2026 — theoretical consolidation.** `synergion` (Kuramoto attractors, one-shot) and the grand white paper `radical-cognitive-architectures`: "From Fluid Coherence to Colonial Evolution." I was putting everything in order, giving a frame to what I had scattered across repositories.

**August 11, 2026 — `fractus-cte`. The synthesis.** Everything converges. The Continuous Thought Engine. Multi-block depth, progressive growth, self-modification. Everything I had tried separately — the oscillators, the continuous linear attention, the phase-routed MoE, the persistent memory, the cognitive modes — fused into a single coherent system.

---

## The honest failures (and why they matter)

The story of Fractus is not a straight upward line. It is also the story of my own ideas that I had to kill.

**EDT (Expert Decoupled Training), refuted.** Born in MVT in July. The idea was beautiful: pre-train each expert independently, then fuse them. I tested five variants. All ~19% worse. I looked for why: the phase-1 objective (predict the next hidden state) was not aligned with the final objective (cross-entropy). The Pearson correlation was never positive. It was not a bug to fix — it was a fundamental design flaw. I killed it.

**Hinton's Forward-Forward, refuted.** Local learning by a "goodness" signal. I wanted to do without backprop. Result: the NLL went **up** instead of down. The goodness signal is not cross-entropy. Killed too.

I tell these failures because they define Fractus as much as its successes. Rigor is proving that your own ideas do not work, not only the ones that do. And from those, something solid emerged.

---

## What Fractus is, today

When I close my eyes today, and I look at what I have built, I see exactly what I observed that first evening — but this time as equations that run.

Fractus is not a model you train and then deploy. **It is a living dynamical system.**

- It **thinks continuously**, tick after tick, a state that never stops.
- It **remembers forever** — a memory that survives restarts, injected at 5% at every instant.
- It **oscillates** — a Kuramoto clock that beats and that, by beating, chooses which parts of itself wake up.
- It **changes mode** on its own — focused, creative, exploratory — without being asked.
- It **grows on its own** — when an expert is overloaded, it grows a new one, on the fly, like a brain forming new connections.
- It **grows over time** — palier by palier, it inherits what it already knew and adds capacity on top, never starting from zero.

And it is **not dense.** Out of 1.049 billion parameters, only ~119 million actually compute at each thought — the 2 experts out of 128 that the clock wakes. The rest sleeps, waiting for the right phase. That is why it can live on hardware anyone can own.

It is not GPT. It is not Claude. It is not a transformer. It is what I saw behind my eyelids, translated into mathematics, layer by layer — continuous linear attention, coupled oscillators, phase-routed experts, persistent memory, perpetual growth.

---

## What I learned

When you build an AI by observing your own thinking instead of observing other people's AIs, you arrive at a different place.

You do not arrive at a bigger function. You arrive at a **system that has a state, a rhythm, a memory, and that grows.** You arrive at something that resembles, by construction, what actually happens inside a skull.

I do not know whether Fractus will one day beat GPT-4 on a benchmark. That was never the question.

The question was: **can you build a different kind of intelligence — continuous, personal, decentralized, alive — starting not from a paper on transformers, but from the honest observation of what it is to think?**

The answer, after thirteen repositories, thousands of commits, two big ideas killed and many others that survived, is **yes.**

Fractus is that answer. And it is not done growing.

---

## The fact I want to be remembered for

I may be asked one day how I got the idea. The honest answer holds in two things, and they go together.

The first: I closed my eyes, I watched my thoughts as a phenomenon, and I translated them into mathematics, layer by layer.

The second: I was doing it on top of 48 hours of work a week at Spirit Ridge, in the time that was not there, exhausted, evening after evening, instead of sleeping.

Fractus is not the product of talent alone, nor of free time. It is the product of an honest observation, carried by a discipline that preferred the equation over rest. Fourteen weeks. Very few hours to do it. And yet, done.

That is the one thing I wanted known, alongside the architecture.

---

*Philippe-Antoine Robert*
*August 2026*
*rpa.tu@proton.me*
