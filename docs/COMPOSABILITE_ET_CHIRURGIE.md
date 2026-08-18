# Proprietes emergentes de Fractus — Chirurgie in-training et composition de checkpoints

**Mis a jour:** 2026-08-17 00:26 UTC

Decouvert en production en corrigeant Fractus pendant l entrainement.

## 1. Chirurgie in-training
On peut modifier Fractus **pendant** l entrainement sans jeter les poids:
- debloquer Kuramoto, brancher le load-balance, changer la temperature des gates
- passer en CE dense, scheduled sampling, recalibrer le LR
- reprendre au **meme offset de tokens**

Regle: garder le .pt, patcher le corps, reload, resume, documenter.

## 2. Plusieurs .pt en parallele puis fusion
On peut entrainer plein de .pt differents (4 GPU / 4 shards) puis les **mean-merge** en un seul Fractus unifie, puis continuer.

## 3. Fusionner un Fractus deja entraine avec d autres .pt
Boucle: train -> merge -> train -> merge ...
Le cerveau s accumule; on ne repart pas de zero.
Deux sens de grossir:
- **composition** (merge de poids meme architecture) — prouve sur ce run
- **croissance structurelle** (paliers, experts, profondeur) — design Fractus

## Limites
Memes shapes obligatoires. Merge trop divergent peut diluer. La gen libre n est pas encore au niveau de la loss teacher-forced.

Voir aussi: docs/COMPOSABILITY_AND_SURGERY.md (version anglaise complete).
