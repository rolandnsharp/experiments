# Speech recognition, round 5: does a bigger model help?

**Date.** 2026-10-24. **Status.** pre-registered as [EXP-0023](https://experiments.blah.dev/EXP-0023) (2026-10-24 09:31 UTC); launch when the RunPod account is topped up.
Prediction written before any run.

## Question

Round 4 ([stt-longer-augmented](../2026-10-23-stt-longer-augmented/)) suggested the 5.8 M
parameter recogniser is short of capacity: 60,000 more updates with augmentation made clean speech
worse (test-clean 0.386 → 0.399 greedy). Round 3 is at 19.4 % WER with the 3-gram on test-clean;
small published streaming models are near 4–8 %. Does accuracy improve with size at a fixed
budget of 30,000 updates, and what does the size cost in speed on the uConsole's A53 cores?

## Prior art

- Neural scaling laws ([Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)): loss falls as a
  power of parameter count while data is not the limit; 886 h is far from limiting at these sizes.
- For speech, Conformer CTC models from 10 M to 100 M+ parameters on LibriSpeech
  ([Gulati et al. 2020](https://arxiv.org/abs/2005.08100)) improve with size, with diminishing
  returns; small models gain the most per parameter.
- What is open: whether this oscillator backbone (complex poles, 50 Hz) scales the same way, and
  whether depth plus MLP width is a good way to spend parameters for it.

## Hypothesis and prediction

At 30,000 updates from scratch, osc front, no augmentation, the recipe of round 3:

| model | params | test-clean greedy | test-clean + 3-gram (beam 64, dev-tuned) |
|---|---|---|---|
| round 3: 4 blocks, MLP 448 (measured) | 5.8 M | 0.386 | 0.194 |
| 6 blocks, MLP 672 | 10.4 M | **0.31–0.35** | **0.15–0.18** |
| 8 blocks, MLP 896 | 16.3 M | **0.26–0.31** | **0.12–0.155** |

test-other moves in the same order, about 1.5x the clean WER.

What would change our mind: 16.3 M greedy worse than 0.35 means capacity is not the main limit
(look at optimisation, the schedule or the architecture instead); better than 0.25 means keep
scaling. If 10.4 M and 16.3 M land within the seed noise of each other, depth stopped paying
(the seed-1 rerun of round 3 measures that noise).

## Independent variable

Model size, along one recipe: width fixed at 448 (harmonics 32), depth and MLP width grow
together (4/448 → 6/672 → 8/896). Depth and MLP width are not separated; the paper reports size
only. Held fixed: osc front (64 poles), data (LibriSpeech 960 h, 886 h after filtering), 30,000
updates, batch 240 s, the optimizer and its schedule, seed 0, no augmentation.

## Speed cost, measured before training

Forward pass, int8, chunk 16, greedy, one A53 core (uConsole core 0), 12 test-clean files
(86.85 s), random weights of each shape (round 3's trained weights time the same as random ones:
0.245 vs 0.233). `bench-a53.sh`.

| model | params | A53 RTF |
|---|---|---|
| 4 blocks, MLP 448 | 5.8 M | 0.233 |
| width 560, 4 blocks, MLP 560 | 9.0 M | 0.267 |
| 6 blocks, MLP 672 | 10.4 M | 0.305 |
| 8 blocks, MLP 896 | 16.3 M | 0.506 |
| width 672, 6 blocks, MLP 672 | 19.3 M | 0.537 |
| sherpa-onnx streaming Zipformer 20 M, int8 (same core, earlier) | 20 M | 0.500 |

The 16.3 M model ties sherpa's speed on the slow cores, so it only stays interesting if its
accuracy comes close to sherpa's.

## Code

`harness/stt/zara_stt.py` (`--layers`, `--mlp`; peak GPU memory now logged),
`pod-run-all.sh` (`LAYERS`, `HARMONICS`), `pod-stt6.sh` (one pod per `TAG`, disk 100 GB, a
watchdog that stops a pod whose logs and disk use stop changing for an hour), `tune-model.sh`.

## Hardware and cost

Planned: two A40 pods ($0.59/h) in parallel, about 1 h of data preparation each, then roughly
1.2 s/update (10.4 M) and 2 s/update (16.3 M): ≈ $7 + $11. A guess from the A53 timing ratios.

## Data, Results, Decided vs measured, Verdict, What it changes

Pending.
