# Speech recognition: a larger pruned 3-gram for the uConsole decoder

**Date.** 2026-10-29. **Status.** concluded, [EXP-0042](https://experiments.blah.dev/EXP-0042):
**refuted** (no gain). Pre-registered 2026-10-28 21:49 UTC, before the run.

## Question

With round 3's acoustic outputs fixed, does the LibriSpeech 3-gram pruned at 1e-7 (2.5 M bigrams,
1.1 M trigrams, 56 MB) beat the 3e-7 one we deploy (1.0 M / 0.34 M, 24 MB)?

## Prediction

test-clean WER at beam 64: 0.194 → 0.180–0.190. Refuted if the gain is under 0.004 or over 0.02.

## Method

Round 3's int8 C-port log-probabilities (dev-clean first 300, all 2,620 test-clean) decoded with
each LM; alpha × beta grid (3 × 4) on dev at beam 32, best pair scored once on test at beam 32
and 64 (`run.sh`, the `tune-model.sh` protocol). Decode timing on the uConsole, one A53 core,
12 test-clean files (86.85 s), from saved log-probabilities, including LM load.

## Results

| | 3e-7 (control) | 1e-7 |
|---|---|---|
| dev-clean 300, best (beam 32) | 0.221 (alpha 0.5, beta 1.0) | 0.215 (alpha 0.5, beta 2.0) |
| test-clean 2620, beam 32 | 0.203 | 0.201 |
| test-clean 2620, beam 64 | **0.194** (CER 0.081) | **0.193** (CER 0.078) |
| LM file | 23.9 MB | 56.4 MB |
| peak RSS, beam 64 (lab) | 67 MB | 99 MB |
| uConsole A53 LM decode RTF | 0.052 | 0.054 |

The control reproduced the 2026-10-23 numbers exactly. Full grids: `tune-3e-7.txt`, `tune-1e-7.txt`.

## Verdict

Refuted: a gap of 0.001, under the 0.004 that counts. The deployed LM stays at 3e-7. The
acoustic model is the limit, not the LM; recheck on round 6's stronger model.
