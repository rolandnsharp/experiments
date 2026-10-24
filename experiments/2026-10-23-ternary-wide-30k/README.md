# Does the wide ternary model's lead grow with training? Ternary vs float from 3K to 30K updates

**Date.** 2026-10-23. **Status.** done (seed 0). Written at 09:50 UTC,
before seed 1 of ternary-projections' arm C landed, gated on that seed
holding. It held (1.9832 vs 1.9818, still ahead of float 4x's 2.0039), and
Roland approved the launch at 10:27 UTC (MARKS `tw30k-t3`, `tw30k-f30`,
`tw30k-t30`). All three arms finished by 20:56 UTC.

## Question

The wide ternary model (4L/64H, 13.2 M ternary-projection parameters, 3.9 MB)
beat the best float 4L/32H model (6.77 MB) by 0.034 bpb at 3K updates (98 MB
of text). Over a full pass of shards 0–1 (30K updates, 983 MB), does that
lead grow, hold or vanish?

## Prior art

- [Spectra 1.1](https://doi.org/10.48550/arxiv.2506.23025) (Vaidhya et al.
  2025): ternary language models gain more from training data than from
  parameters, according to their scaling-law fit. Measured at 99 M+ and on
  BPE tokens; open at 13 M on bytes.
- [Spectra](https://doi.org/10.48550/arxiv.2407.12327) (Kaushal et al. 2024):
  at equal bits, ternary beats float above about 1 B parameters. Our 3K
  result is already an equal-bytes win at 13 M. Whether it holds with more
  data is the open part.
- [Björck et al. 2024](https://doi.org/10.48550/arxiv.2409.19913): the best
  learning rate falls as training tokens grow. So 4x, best at 3K, is
  probably too high at 30K. Both arms here run at 2x, a compromise stated as
  such.
- In this repo: training-bytes-vs-bpb-and-core (float 4L/32H, 1x: 2.0490 at
  3K → 1.8469 at 30K, still falling) and learning-rate-4l32h (float polar
  gains 0.033 from 1x to 2x at 3K).

## Hypothesis and prediction

Written 2026-10-23 09:50 UTC, before any arm below:

- **T3** (wide ternary, 3K, lr 2x): 1.98–2.00 bpb, within 0.02 of the 4x
  arm's 1.9818.
- **F30** (float 4L/32H, 30K, lr 2x): 1.825–1.845. Gains 0.00–0.02 over the
  1x run's 1.8469, since the rate gain shrinks with a longer horizon.
- **T30** (wide ternary, 30K, lr 2x): 1.76–1.80. **Its lead over F30 grows
  from about 0.03 at 3K to 0.04–0.08 at 30K.**

What would change our mind:
- If the lead at 30K is under 0.02 or reversed, ternary's 3K advantage was
  an early-training effect (fast early fitting from the larger width), and
  the edge line has to rest on bytes alone, not quality.
- If the lead grows past 0.10, the result says ternary width is a cheap way
  to buy quality at fixed bytes, and the next step is a size ladder.

## Independent variable

Training length, 3K → 30K updates (98 MB → 983 MB, one pass over shards
0–1), crossed with the model:
- float 4L/32H (polar, dense, 3.39 M parameters);
- ternary 4L/64H (polar, dense, `--ternary`, 13.19 M parameters).

Held fixed: lr 2x (`--lr-scale 2`) in every arm, batch 16 x 2048, warmup 200
and cosine to 10% over each arm's own length, seed 0, the same shard order,
eval every 500 on 256 held-out windows, the final score over all held-out
bytes.

The comparison of interest is the ternary − float gap at 3K against the
same gap at 30K. F3 (2.0157) already exists. T3 is new because the 3K wide
arm ran at 4x, not 2x.

## Code

`harness/field_poles.py train --variant polar --mixing dense --layers 4
--harmonics 32|64 --batch 16 --updates 3000|30000 --seed 0 --shards 2
--lr-scale 2 [--ternary]`, as committed in 7f9fa5a. Pods via `pod-self.sh` on
Roland's account, one launcher per arm.

## Hardware and cost

Community RTX 3090 at $0.22/h. Estimated from the 3K runs' step times:
- T3: about 75 min, $0.30;
- F30: 30,000 x 0.87 s ≈ 7.3 h plus scoring, about $1.70;
- T30: 30,000 x 1.16 s ≈ 9.7 h plus scoring, about $2.30.

About $4.30 in all. The arms run in parallel.

## Data

OpenWebText shards 0–1 as bytes, 98,304,000 bytes seen at 3K and 983,040,000
at 30K (one pass). Held out: the GPT-2 baseline's split (sha256 d0684974…,
18,538,212 scored bytes). The CORE bundle is never in training.

## Results

Held-out bpb over all 18,538,212 scored bytes, seed 0, lr 2x:

| updates (bytes) | float 4L/32H (6.77 MB) | ternary 4L/64H (3.90 MB) | ternary − float |
|---|---|---|---|
| 3K (98 MB) | 2.0157 (F3, learning-rate-4l32h) | **1.9897** (T3) | **−0.026** |
| 30K (983 MB) | **1.8396** (F30) | **1.8344** (T30) | **−0.005** |

Subset curves (256 windows) inside the 30K runs, float / ternary / gap:
3K: 2.0557 / 2.0369 / −0.019; 6K: 1.9862 / 1.9811 / −0.005; 9K: 1.9306 /
1.9289 / −0.002; 12K: 1.9040 / 1.9052 / +0.001; 15K: 1.8928 / 1.8907 /
−0.002; 18K: 1.8770 / 1.8837 / +0.007; 21K: 1.8620 / 1.8619 / 0.000; 24K:
1.8376 / 1.8327 / −0.005; 27K: 1.8242 / 1.8168 / −0.007; 30K: 1.8181 /
1.8100 / −0.008. The lead is gone by 6K and the two run level from there,
ternary a hair ahead at the end.

F30 against the 1x float 30K run (training-bytes-vs-bpb-and-core, 1.8469):
−0.007 from the 2x rate at 30K, against −0.033 at 3K. The rate gain shrinks
with the horizon, as Björck et al. predict.

Compute: F30 3.62 M MAC/byte, 2.14e16 training FLOP, 23,272 s (pod
7zld3vjdkxy14k); T30 13.64 M MAC/byte (96% ternary adds), 8.05e16 FLOP,
35,064 s (pod 23h9qch0yt0q3i). T3: 1.9897 bpb, pod 6h18pkum3ijcs3 (RTX
3070). About $4 in all.

Report beside bpb: model bytes, multiply-adds per byte (true multiplies vs
ternary adds), training FLOP, wall-clock, cost; the subset curve for both
30K arms; CORE smoke for T30 and F30 if the endpoint can serve a ternary
checkpoint (`core_server.py` would need `--ternary`; it does not yet).

## Decided vs measured

Decided: the ternary statements on proofs.blah.dev (see ternary-projections).
Measured: every bpb and CORE number, one seed.

## Verdict

Seed 0. The prediction held for T3 (1.9897, predicted 1.98–2.00) and for
F30 (1.8396, predicted 1.825–1.845). It **failed for T30**: 1.8344 against a
predicted 1.76–1.80. **The lead did not grow; it shrank from 0.026 to
0.005**, under the 0.02 we said would change our mind.

So the wide ternary model's 3K advantage was mostly early-training speed:
the larger width fits faster at first, and float catches up within about
6K updates. Over a full pass the two are level in quality. Spectra 1.1's
"ternary gains more from data" did not show here, at 13 M parameters on
bytes.

What survives is the bytes-and-arithmetic story, not a quality one: equal
held-out bpb at 1.7x fewer model bytes (3.90 vs 6.77 MB), with the
projections as integer adds. Against that, it does 3.8x the multiply-adds a
byte and took 1.5x the training wall-clock.

Caveats: one seed at 30K; one model size; lr 2x for both arms, which may
not be either one's best at 30K.

## What it changes

If the lead grows, the next steps are 8-bit activations, then a C integer
port timed on the uConsole, and a ternary arm at Diakosmos's size. If it
vanishes, the ternary line is a bytes story only, and the poles (whose gain
grew to 0.10 at 100 M, diakosmos-size-pole at 640 updates) are the stronger
lead.
