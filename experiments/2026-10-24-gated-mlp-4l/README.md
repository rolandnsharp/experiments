# A gated MLP in the polar block, at equal parameters

**Record.** [EXP-0024](https://experiments.blah.dev/EXP-0024) on experiments.blah.dev; the numbers there and here match. This protocol was committed to git before launch while the site was unreachable; the site record was created from it and pre-registered at 10:02–10:04 UTC, and its methodology states which runs had started by then.

**Date.** 2026-10-24. **Status.** concluded 2026-10-24: supported in direction, gain larger than predicted.

## Question

Diakosmos's block, and so Vidya's, has no feed-forward sublayer: each block is the temporal
recurrence and a spatial mix, joined by a gate. Does adding a SwiGLU feed-forward sublayer
help, when the parameters it costs are taken from the recurrence's harmonics?

## Prior art

- GLU variants ([Shazeer 2020](https://arxiv.org/abs/2002.05202)): gated feed-forward
  sublayers (SwiGLU) beat ReLU/GELU ones in Transformers.
- Hawk and Griffin ([De et al. 2024](https://arxiv.org/abs/2402.19427)) interleave MLPs with
  LRU-style recurrent blocks; most recurrent LMs (RetNet, GLA, Hawk) keep an MLP after the
  mixer. Diakosmos does not. Borrowed: the SwiGLU sublayer. Ours: the measurement in this
  pole recurrence at equal parameters.
- text-pole-form (this repo): polar 4L/32H at 3K updates, lr 1x: 2.0490 (seed 0), 2.0409
  (seed 1).

## Hypothesis and predictions

Written before any run.

1. The arm with the MLP scores **0.01–0.05 bpb below** the control, as the mean over seeds 0
   and 1, and below it on each seed. A mean gain under 0.01, or the MLP arm worse on either
   seed, refutes it: then the harmonics are worth more than the MLP at this size.
2. Its step time is within 1.3x the control's (the MLP is dense products, cheap on a GPU).

## Variables

- Independent: block form. Control: polar 4L/32H (3,385,688 parameters). Treatment: polar
  4L/20H plus a SwiGLU sublayer of width 600 after each block, `--mlp 600`
  (`z + fc2(silu(fc1 LN z) * fc3 LN z)`), 3,383,272 parameters (-0.07%).
- Held fixed: 4 layers, Diakosmos's optimizer and schedule for 3,000 updates at his peak
  rate (lr 1x; fc1/fc2/fc3 in the rotational group, the MLP's layer norm in Adam), batch 16,
  2 shards, the held-out set, seeds 0 and 1.
- Measured: held-out bpb over all 18,538,212 scored bytes; step time; multiply-adds a byte.

## Code

`harness/zara_poles.py` and `harness/field_poles.py`, new flags `--mlp F` (and `--select`,
the sister experiment), off by default. Checked on the lab's CPU before use: without the
flags the run is bit for bit the previous harness's (`dev/test_harness.sh`); the chunked
forward equals the position-by-position sampler with the MLP on (1.7e-5); the new
parameters are drawn from their own generator after every other parameter, so every
pre-existing parameter starts identical; a 3-update training smoke ran.

## Runs and data

OpenWebText bytes as text-pole-form (`harness/owt-manifest.json`), first 2 shards; the
held-out set is the GPT-2 baseline's windows, never trained on. One community RTX 3090/3080 on
Roland's RunPod account runs both seeds in sequence: `LEGS="polar:dense:3000:0
polar:dense:3000:1" LAYERS=4 HARMONICS=20 BATCH=16 SHARDS=2 EXTRA="--mlp 600" TAG=mlp600`.
The control is not rerun: its numbers come from the same harness, bit-identical with the
flags off.

## Analysis

Gain = control - treatment per seed; the verdict is on the mean and both signs. Two seeds
at 3K is a screen: a pass earns a run at more updates and the 6L/100M sizes, not a claim.

## Results

Held-out bpb over all 18,538,212 scored bytes after 3,000 updates:

| seed | control (h32) | h20 + MLP 600 | gain |
|---|---|---|---|
| 0 | 2.0490 | 1.9309 | 0.118 |
| 1 | 2.0409 | 1.9303 | 0.111 |

Mean gain **0.114**, seed spread of the MLP arm 0.0006; the gap is 0.11–0.14 from the first
evaluation (subset curve, control seed 0 / MLP seed 0 / seed 1: 500: 2.447 / 2.310 / 2.316; 1500:
2.119 / 2.009 / 2.008; 3000: 2.028 / 1.905 / 1.906). Compute: 1.52 M MAC/byte against 3.62 M (the
20-harmonic model is narrower, width 280 vs 448); 0.60–0.68 s an update against 0.75 s on community
3090s; $0.13 + $0.11.

## Verdict

Prediction 1 missed on the high side: the gain is more than twice the top of the predicted
0.01–0.05, on both seeds (refutation condition, a gain under 0.01 or a loss on either seed, not met).
Prediction 2 met: 0.8x the step time. The MLP and the narrower width travel together here;
mlp-split-4l (EXP-0030) sweeps the split, factorial-6-4l (EXP-0029) its interactions. The block
taken to Diakosmos's size should carry an MLP; the 100M polar-only run was held for this reason.
