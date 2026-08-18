# Decode Surgery Log — Breaking Generation Attractors

**Date:** 2026-08-16 19:31 UTC

## Diagnosis (confirmed)

- Train CE can be very low (GPU1 ~2.7 under stage2 dense CE) while greedy/default decode collapses.
- Knowledge is in the weights (force-pos2 test: diversity appears when loop is broken).
- Failure mode is **decode plumbing**: thought_state + Kuramoto + autoregressive feedback converge to a fixed point or short trajectory attractor.

## Surgeries applied (all decode-time, weights unchanged)

| # | Surgery | Effect |
|---|---------|--------|
| 1 | Soft gates at decode (temp 2.5-3.5) | Wider expert mix |
| 2 | Phase noise each tick | Breaks phase fixed point |
| 3 | Thought-state noise + decay | Breaks residual attractor |
| 4 | Ban recent token window | Stops single-token lock |
| 5 | Frequency penalty | Limits token overuse |
| 6 | Cycle detect period 1-20 + hard scramble | Targets short loops |
| 7 | Ban repeating block after cycle | Extends cycle break |
| 8 | Periodic forced escape token (every 4) | Breaks multi-token trajectory attractors |
| 9 | Attn state zero on hard scramble | Clears attention carry lock |

Module: 

## Results summary

| Mode | Unique tokens / 40 | max_freq | Notes |
|------|-------------------|----------|-------|
| Baseline | **1** | 40 | Pure single-token collapse |
| Phase+ban only | ~9-11 | high | Multi-word short cycles |
| Cycle20 + scramble | ~14-21 | 4 | Near-repeat phrases |
| Full + escape every 4 | **31-34** | **2** | Cycles broken; not yet coherent language |

## Full outputs (radical escape stack)

### The meaning of life is

- unique: 31 | max_freq: 2

     uphe Chosen Surely prize Jo �obi Cosmetic convinceINTON numbering wontKill mah sinners concussionpayer conceived mattBOOK Denver uphe Chosen Fate Such prize � Tercle TowersINTONormonsrepe subreddit mah sinnersundown Pens conceived matt

### Hello, my name is

- unique: 32 | max_freq: 2

     Colorado waist operating. Medicare 2005 adorable Deadline Uberissors MongoliaFer discussed grandchildren Logged
Jean recognizedftenioticsFran Lenn waistatlanticaux. archetype adorable layingpositiveissorsliberalibelFer labor Logged mathematic
 Heathften

### Once upon a time

- unique: 34 | max_freq: 2

     commands desperate Punk coverage TuOilvalues upcomingfruit Cat adjustmentsadh ownership combustion bos Lovalty Adelaide comprisesund ColonyRepl cheers Torrent Civilization hierarchical Herevalues teenage surgeonsjury1998 guidelines Cat combustion bososen participation Adelaide comprises

### The capital of France is

- unique: 32 | max_freq: 2

     Chosen Surely uphe ._ occurrence prize sinners TowersfetINTON mah � boothetime sailsceansSoon conceived Pen matt sket Chosen Weightfighting Leia Assignment Gil sinnersINT TowersINTON Bian cardio mahetime Along wrapped Pens conceived sails

### Fractus thinks

- unique: 32 | max_freq: 2

     Population 420 sle Russohas Diff ensuing opaqueibauable swallowing Gerald bonus Maburn Greater immoraltones measured determines room Dwarf 420 derail Bever again Diff Accessed perl ensuinguable swallowing PJ Gerald surgeons Muse icy Greatertones hyd

### What is consciousness?

- unique: 32 | max_freq: 2

     Towers origfax MP targets Fateintention Britann 291 involve separatelygrass merchants Sharia Along 303 canoe Pens vanish chance variability TowersFightfax Apache Trade Fate Chosen Lup Britann Daneki tragedyraitsintention Alongasio Kim Pens vanish

## What is settled

1. Single-token lock: **fixed** at decode with surgery.
2. Short multi-token cycles: **mostly fixed** with escape injections (max_freq 2, unique >30).
3. Coherent sentence generation: **not fixed** — needs more structured learning / better decode alignment, not just noise.
4. Training continues stage2 dense CE on 4 GPUs; decode surgery does not require weight changes.

## Open next (optional)

- Learned decode controller (when to escape) instead of random forced tokens
- Train with explicit anti-repetition / contrastive sequence objectives
- Memory injection at decode time from PersistentMemory
