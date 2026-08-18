# Stage2 SS Generation Probe

**Date:** 2026-08-16 23:55 UTC

**Checkpoint:** checkpoints/fractus_1b_gpu1.pt (SS phase, loaded 391 tensors)

**Train snapshot at probe:** GPU1 ~200.4M tokens, tf~1.4, ema_tf~1.35

## Method

- generate_chunk: continuous tick_chunk path
- generate_window: re-encode last 32 tokens each step
- temperature 0.8, max_new 28

## Full outputs

loaded 391 cfg_keys ['d_model', 'n_heads', 'd_head', 'n_levels', 'n_oscillators', 'coupling_rank', 'n_experts', 'top_k', 'expert_d_ff', 'siren_rank', 'n_layers', 'gpu']
PROMPT: The capital of France is
  CHUNK u=10 'vers31cash Quest Initi186As maywikipediavers31cash Quest Initi186As maywikipediavers Stanfordcash Quest Initi186As maywikipediavers'
  WINDOW u=25 'vers limit artisanrived Phill Creifled margins:( Betsycluded rehearsaludastorms ExplorerivedPhiladelphia QuestExternal TitusEd Cre fortTesting Cecrived legal Road'
PROMPT: Hello, my name is
  CHUNK u=11 ' Exploreartasteen actorPureweet Benghazi clickeddad Exploreartaular actorPureweet Benghazi clickeddad fossartaular actorPureweet Benghazi clickeddad foss'
  WINDOW u=28 ' Exploresteen 315 */ randitude treadmill developmentawan Reserveadderumni Across513Kargd ApproBlackandoAI shader limbo counterproductiveTumblrifications bends prevalent subtly'
PROMPT: Once upon a time
  CHUNK u=12 ' Guyizzle prolongjeej hilar flirt appointmentsBreaking vaultaden prolongjeej hilar flirt appointmentsBreaking vaultaden choppingjeej hilar flirt appointmentsBreaking vault'
  WINDOW u=25 ' Guy DT impressiveproofmist vault secrets fabricatedWHEREerto BW rehearsalikawa husalks companies investigates TOP beaches regain keeper repealedtrained engineered secrets fabricatedWHERE shifting'
PROMPT: The meaning of life is
  CHUNK u=11 'rivedAs Prosecutorverswikipedia Quest186 impePl mosaicAs Prosecutorverswikipedia Quest186 impePl mosaicAs Prosecutorverswikipedia Quest186 impe??? mosaic'
  WINDOW u=27 'rivedRAM thereLondonLatiesedition epidem sellers Estate GamerRegisterrived Phill FinallyYork Jas malwareVill五ets Evanellig UNIVERSsty anomalyFlagscreat'
PROMPT: Fractus thinks
  CHUNK u=10 ' Meal diss prejud prevalent655 CONS flirt sales wonderful Meal diss prejud prevalent655 generic flirt sales wonderful Meal diss prejud prevalent655 generic flirt sales wonderful Meal'
  WINDOW u=25 ' Meal55 unchangedppo diss relief Summary SourceCVE analogue drunken}}idelppoJUST unchanged dissYo bul Infantry quality volunte reliantiffs photograph thoroughly admoncome'
PROMPT: In mathematics
  CHUNK u=11 ' NinVW Phen eating Sweep maintained ArkansasdirPhill 1934VW Phen eating Sweep maintained ArkansasdirPhill 1934VW Phen Integ Sweep maintained ArkansasdirPhill 1934'
  WINDOW u=28 ' transpludebgun massage OddVWidered communicateAnn house deletioneralaedition genre technologies NEED suburb pistfaneware00200000 Gettyeanor Trial maintained playable Carol Vector'


## Reading

- Still no coherent English sentences
- CHUNK remains cyclic multi-token patterns
- WINDOW has higher uniqueness (~25-28) and less strict loops
- Consistent with exposure-bias diagnosis: TF loss low, free-run text not yet language

