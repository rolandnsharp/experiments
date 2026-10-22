# Index

| date | experiment | variable | result | verdict |
|---|---|---|---|---|
| 2026-10-19 | [speech-pole-form](experiments/2026-10-19-speech-pole-form/) | pole form (real / polar / cartesian) on speech bytes | 3.396 / 3.293 / 3.281 bpb | complex poles win 0.10–0.12 bpb at equal updates (one seed) |
| 2026-10-21 | [text-pole-form](experiments/2026-10-21-text-pole-form/) | pole form (real / polar) on OpenWebText bytes | seed 0: 2.0849 / 2.0490; seed 1: 2.0842 / 2.0409 | polar wins 0.036–0.043 bpb at 3K updates, two seeds; 0.037 at 13.5K (one seed) |
| 2026-10-21 | [grouped-vs-dense-mixing](experiments/2026-10-21-grouped-vs-dense-mixing/) | per-head (block-diagonal) vs dense q/k/v/out projections at equal width | 2.7044 vs 2.0490 bpb | grouped loses 0.65 bpb at equal width (5.3x fewer params); equal-compute comparison untested |
| 2026-10-21 | [training-bytes-vs-bpb-and-core](experiments/2026-10-21-training-bytes-vs-bpb-and-core/) | bytes seen (98 MB → 983 MB, one pass) at 4L/32H | 2.0490 → 1.8469 bpb; CORE smoke 0.004 → 0.044, standard 0.009 → 0.018 | still falling at the end of the pass; data is the cheap lever |
| 2026-10-22 | [width-depth-6l64h](experiments/2026-10-22-width-depth-6l64h/) | model size 4L/32H → 6L/64H (3.4 M → ~20 M params) | running | prediction: 1.60–1.70 bpb, smoke CORE 0.05–0.06 |
