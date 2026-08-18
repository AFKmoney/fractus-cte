# Train-Aligned Generation (100% tick_chunk)

**Date:** 2026-08-16 21:42 UTC

Module: fractus/generate_aligned.py

- generate_chunk: continuous state, length-1 tick_chunk steps
- generate_window: re-encode last N tokens each step (fresh causal window)

No tick_single in either path.

## Results (GPU1 stage2 ckpt, CE~2.1)

### The capital of France is

- CHUNK unique=11: cript reproductive Love Below186 Exchange (). Byz novelwikipedia reproductive Love Below186 Exchange CBC Byz novelwikipedia reproductive Love Below186 Exchange
- WINDOW unique=23: cript Cre TD Civil sob Wilson Xuan ExchangeLa Loveythonizons heaviestdden heavily spiceplex Dip Love Cec fortTestingrived penetrated

### Hello, my name is

- CHUNK unique=10:  Explore weaponfficdk 315 Levantootingо� actor Explore weaponfficdk 315 Levantootingо� actorulf weaponfficdk 315 Levant
- WINDOW unique=21:  Explore snippetitude oldest Fail 104 bounty523 congenIntern hereditary unmreen habitsvid POS valves Cann prepared prevalent meetings unmreen habits

### Once upon a time

- CHUNK unique=11:  Havingerto Byz Harris appointmentsCtrl Caller [| Everest producederto Byz Harris appointmentsCtrl Caller [| Everest producederto Byz HarrisGivenCtrl
- WINDOW unique=21:  Havingerto needy loadercroft impressiveproofmist vault secrets Roth BurnettARPerto enthusiast complications sentenced TMZrimp Arctic husARP Burnett warranties

### The meaning of life is

- CHUNK unique=11: 31#$tiesレ186 impeclaim Prosecutor Love CBC#$tiesレ186 impeclaim Prosecutor Love CBC#$???レ186 impe
- WINDOW unique=23: 31rers thereLondonLaties186ib Stim Phillcat awaiting POSoda Eff Neurologrived nations timer Categories Byz CreOFF POS

### Fractus thinks

- CHUNK unique=10:  Meal diss CONS Kon innings \' chants propos Atmosp Meal diss CONS Kon innings \' chants propos Atmospmber diss CONS Kon innings \'
- WINDOW unique=19:  Meal55 unchanged sacridelppo dependency Summary242 reliantistered NGO sacridel autos analogue diss relief Infantry Sean reliantistered NGOEntity

## Reading

- WINDOW mode breaks cycles better (unique ~19-23 vs chunk ~10-11).
- Still not coherent English sentences.
- Path alignment removes train/gen dynamics mismatch for MoE+Kuramoto+chunk attn.
- Remaining gap is representation quality / more digestion, not mono-token lock.
