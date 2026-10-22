# Training bytes vs held-out bpb and CORE, 4L/32H polar dense (98 MB → 983 MB)

**Date.** 2026-10-21. **Status.** done. Three points at three sizes of bytes
seen; the two 4L/32H points differ only in updates (one variable); the CPU
point is a smaller model and is marked as such.

## Question

With the model fixed, how far does one pass over the 1 GB shard take bpb and
CORE, and is the curve still falling at the end (i.e. is data the cheap lever)?

## Prior art

- Byte Latent Transformer ([Pagnoni et al. 2024](https://arxiv.org/abs/2412.09871v1)):
  byte-level models scale with compute like token models at 1–8 B parameters;
  nothing at 3 M parameters or on OpenWebText.
- Distilling byte models ([arXiv 2609.12303](https://www.alphaxiv.org/abs/2609.12303)):
  byte models overtake token models at higher compute, ~1 B parameters.
- The CORE leaderboard references on evals.blah.dev: GPT-2 124M smoke 0.0796;
  Ajax's Alpha GPT-2 124M (9 B tokens) smoke 0.0610, standard 0.0578.

## Hypothesis and prediction

Written before the 30K run: 1.85–1.90 bpb at the end of the pass; smoke CORE
0.01–0.03. Expected the curve to flatten in the last third.

## Independent variable

Updates (and so bytes seen): 3,000 vs 30,000 at batch 16 x 2048 (98 MB vs 983
MB, one pass, no repetition). Held fixed: polar, dense, 4L/32H, seed 0,
optimizer, schedule (cosine to the run's own length — the schedule therefore
moves with the variable). The CPU point (3L/16H, 684 K params, 2000 x 8 x
2048 = 33 MB) is a different model and only anchors the CORE line.

## Code

`harness/field_poles.py` as in text-pole-form; CORE served by
`harness/core_server.py` (blah-core-http/1, byte vocab, context 2048, the
answer span scored from one byte before the shortest option ends);
registration `harness/register-vidya.sh`.

## Hardware and cost

3K: secure 3090 ($0.50/h), 40 min. 30K: secure 3090 ($0.50/h, pod
58n98ot8n3lu73), 7.3 h ≈ $3.65. CPU point: 6 lab threads, 4.0 h. CORE scoring
on 4 lab CPU threads: smoke ~14 min, standard ~2 h.

## Data

OpenWebText shards 0–1 (1,001,644,862 bytes). Held-out 18,538,212 scored
bytes (the GPT-2 baseline's windows). Eval bundle never in training.

## Results

| checkpoint | params | bytes seen | held-out bpb | CORE smoke | CORE standard | MAC/byte | training FLOP |
|---|---|---|---|---|---|---|---|
| cpu-polar 3L/16H | 684,482 | 33 MB | 2.3583 | −0.0138 | — | 0.78 M | 1.5e14 |
| polar-dense 3K | 3,385,688 | 98 MB | 2.0490 | 0.0044 | 0.0089 | 3.62 M | 2.14e15 |
| polar-dense 30K | 3,385,688 | 983 MB | **1.8469** | **0.0436** | **0.0182** | 3.62 M | 2.14e16 |

Reference: GPT-2 124M, 32 M MAC/byte, 6.7e18 FLOP on 9 B tokens: smoke 0.0796;
Ajax's Alpha GPT-2: smoke 0.0610, standard 0.0578.

30K curve (256 windows): 1.5K 2.124, 3K 2.068, 6K 1.986, 10K 1.924, 15K 1.890,
20K 1.855, 25K 1.833, 28.5K 1.826, 30K 1.824. Slope over the last 5K: −0.009
per 5K updates, not flat.

CORE per task at 30K (standard, centred): commonsense_qa 0.275,
language_identification 0.162, arc_easy 0.085, operators 0.076, copa 0.06,
dyck 0.056, winogrande 0.032; boolq −0.40 (it answers "yes" to everything);
the generation tasks (jeopardy, squad, coqa, lambada, cs_algorithms,
repeat_copy, qa_wikidata) 0. Smoke's higher number comes from hellaswag 0.147
on 50 items (0.003 on 500) and boolq at 0 instead of −0.4: the smoke tier is
noisy at this level.

Samples at 30K read as English fragments ("including his clap were infringer
and terminated this movement"); at 3K, word-shaped gibberish.

## Verdict

bpb 1.8469 beat the predicted 1.85–1.90 by a hair; smoke CORE 0.0436 beat the
predicted 0.01–0.03. The curve was still falling at the end of the pass, so at
this size the model is data-bound: more bytes are the cheap lever. Standard
CORE (0.018) is the number to trust; smoke overstates it.

## What it changes

Prepared shards 2–9 (10 shards, 5 GB on the lab) and queued the 6L/64H run
(width-depth-6l64h), which moves model size and bytes together and is
therefore not a clean single-variable result; the clean follow-ups are
4L/32H on 3 GB (data only) and 6L/64H on 983 MB (size only).
