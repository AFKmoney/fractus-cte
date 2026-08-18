# Notes d'entrainement Fractus

**Mis a jour:** 2026-08-18 00:17 UTC

Procedure complete (corpus HF, phase1/phase2, 8 GPU, SS, merge, recovery):

-> **HOW_FRACTUS_IS_TRAINED.md**

### Rappel court

1. Source de verite corpus = thefinalboss/fractus-datasets
2. Phase2 = tokenization complete du raw -> tokenized/phase2/shard_phase2_gpu*.npy (~3.44B tokens)
3. Train = 1 process / GPU, tick_chunk_train + scheduled sampling, checkpoints fractus_1b_gpu{i}.pt
4. Merge moyen des 8 cerveaux + push HF (Xet)
5. Resume via RESUME_MANIFEST_8GPU.json (offsets)

Cap maintenu: phase2 en cours sur 8x5090.
