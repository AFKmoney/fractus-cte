# Ops while away — 2026-08-17T21:01:43.997574+00:00

## Running now
- 8x GPU B=3 SS train on **1.52B tokenized corpus** (all .pt available on HF dataset)
- shard_len per GPU ≈ **190,125,756** tokens
- Progress tracker: `/workspace/progress_tracker.py` → `checkpoints/RESUME_MANIFEST_8GPU.json` every 2 min
- Hourly HF Xet sync of 8 ckpts + merge (daemon)

## Anti re-ingest
- Manifest stores `start_token_next` per GPU from live logs
- On any restart: use `START_TOKEN` from manifest, never 0 if progress > 0
- Phase2 job tokenizes raw jsonl/gz (cognitive_skills, neuro_code_math, paradigms, repos...) into **new** `shard_phase2_gpu*.pt` — separate stream for after phase1

## Missing 4.23B
- `full_corpus.pt` (4.23B) was **never on HF** — only referenced in scripts; lived on dead pod
- Recoverable tokenized on HF now: **~1.52B**
- Phase2 raw tokenize aims to add remaining text not fully covered by those .pt files

## When phase1 ends (~40h @ 1300 tok/s)
1. Read RESUME_MANIFEST
2. Switch SHARD to `shard_phase2_gpu{i}.pt` START_TOKEN=0
3. Or concat phase2 onto end of phase1 shards if preferred

## Do not
- Destroy pod
- Restart from START_TOKEN=0 without reading manifest
