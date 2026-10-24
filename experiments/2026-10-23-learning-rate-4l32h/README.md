# Peak learning rate at 4L/32H: is Diakosmos's 0.003 too low, and does the pole gain survive a tuned rate?

**Date.** 2026-10-23. **Status.** done (one seed, 3K updates). The paper and prediction were
written at 08:35 UTC by the ternary-projections session, before any arm of
the grid below had a result except polar 2x (see "Known before writing").
The arms are run by the Vidya session (launchers `lrcheck-*` on Roland's
account), which fills in the results.

## Question

Is the peak learning rate every 4L/32H experiment inherits (Diakosmos's
`PEAK_LR` 0.003) too low for this model size at 3K updates? And does the
pole result (polar beats real by 0.036–0.043 bpb, text-pole-form) still hold
when each pole form gets its own best rate?

## Prior art

- `PEAK_LR` 0.003 and `MUON_RATIO` come from harmonic-gpt's 100 M settings
  (`diakosmos_oracle.py`), tuned for a model about 30x larger than 4L/32H
  (3.39 M parameters) and a 15,360-update schedule. Smaller models usually
  take higher rates, so 0.003 was never expected to be right here; nobody
  had varied it.
- [Björck et al. 2024, Scaling Optimal LR Across Token Horizons](https://doi.org/10.48550/arxiv.2409.19913):
  the best learning rate falls as the number of training tokens grows. So a
  rate tuned at 3K updates (98 MB) is likely too high for the 30K full pass
  (983 MB). This experiment settles the 3K comparisons; it does not settle
  the 30K headline.
- [Power Scheduler](https://doi.org/10.48550/arxiv.2408.13359) (Shen et al.
  2024): the best rate depends jointly on batch size, tokens and model size.
  This argues for tuning at each size we report, not inheriting a rate.

## Known before writing

One arm had landed when this was written. Polar at 2x (`--lr-scale 2`),
seed 0, ran as the fairness check in ternary-projections and scored
**2.0157** bpb (control at 1x: 2.0490). That result is what prompted this
experiment, so it is not a prediction. The predictions below are for the
three arms that had not landed: real 2x, real 4x and polar 4x.

## Hypothesis and prediction

Written 2026-10-23 08:35 UTC, before real 2x, real 4x and polar 4x.

- **Real decay gains at least as much as polar from a higher rate.** Real
  2x and 4x: 2.03–2.06 bpb (real 1x: 2.0849).
- **Polar 4x is within ±0.015 of polar 2x** (2.00–2.03): the best rate lies
  at 2–4x for both forms.
- **The pole gain at each form's best rate is 0.01–0.04 bpb.** It shrinks
  from 0.036–0.043 but does not vanish.

What would change our mind:
- If the gain at the best rates is under 0.01, the pole result was mostly
  the poles tolerating a bad rate, and the headline claim (text-pole-form,
  the reason Vidya differs from Diakosmos) needs rewording.
- If real gains much less than polar from the rate, the pole gain grows, and
  the 1x result understated it.

## Independent variable

The peak learning rate, through `field_poles.py --lr-scale S` (S = 1, 2, 4).
It multiplies the oracle's `PEAK_LR`, so Adam's and Muon's rates scale by S
and the rotational step's by sqrt(S). Crossed with the pole form (real,
polar).

Held fixed, as text-pole-form: 4 layers, 32 harmonics, dense mixing, batch
16 x 2048, 3000 updates, warmup 200 and cosine to 10%, seed 0, the same
shard order, the same scoring.

The 1x arms are text-pole-form's seed-0 runs and are not rerun.

## Code

`harness/field_poles.py train --variant real|polar --mixing dense --layers 4
--harmonics 32 --batch 16 --updates 3000 --seed 0 --shards 2 --lr-scale S`.
`--lr-scale` was added in commit 7f9fa5a; with S = 1 the run is bit for bit
the old harness (`harness/ternary_check.py`).

## Hardware and cost

Community RTX 3090/4090/3080 on Roland's account, about 40 min an arm at
$0.22/h. Polar 2x: pod 87bthpz2cugrsj, 07:30–08:10 UTC. Real 2x: pod
6pjs8ct6mv5rrt from 08:28 UTC. Real 4x and polar 4x: waiting for stock at
writing.

## Data

As text-pole-form: OpenWebText shards 0–1 as bytes, 98,304,000 bytes seen
per arm. Held out: the GPT-2 baseline's split (sha256 d0684974…,
18,538,212 scored bytes). The CORE bundle is never in training.

## Results

Full held-out (all 18,538,212 scored bytes), 4L/32H, 3,000 updates, seed 0.
The 1x arms are text-pole-form's; polar 2x is the ternary session's
float-lr2 run; the rest ran on Roland's account (community RTX 3090s,
launchers `lrcheck-*`).

| peak rate | real | polar | gap (real − polar) |
|---|---|---|---|
| 1x (0.003) | 2.0849 | 2.0490 | 0.036 |
| 2x | 2.0519 | 2.0157 | 0.036 |
| 4x | 2.0381 | 2.0039 | 0.034 |
| 8x | 2.0360 | 1.9992 | 0.037 |

Real 2x: pod 6pjs8ct6mv5rrt, 1,888 s; real 4x: 1,723 s; polar 4x: pod
3se6eb4flh8diw, 09:31–10:20 UTC.

## Decided vs measured

Decided: nothing new. Checked: with S = 1 the run is bit for bit the old
harness. Measured: every bpb number, one seed.

## Verdict

All three predictions held, on one seed at 3K updates:

- Real decay gained at least as much as polar from a higher rate: 2.0519 at
  2x and 2.0381 at 4x, inside the predicted 2.03–2.06 (1x to 4x: −0.047;
  polar 1x to 4x: −0.045).
- Polar 4x (2.0039) is within the predicted ±0.015 of polar 2x (2.0157).
- The pole gain at each form's best rate tried (4x for both) is **0.034**,
  inside the predicted 0.01–0.04, and it is 0.036, 0.036, 0.034 at 1x, 2x
  and 4x: **the pole result is not an artefact of the learning rate.**

The 8x pair (Roland's account, community 3090s, 2026-10-23 evening): real
2.0360, polar 1.9992, a gain of 0.037. Both forms are flat from 4x to 8x
(−0.002 and −0.005), so the best rate at 3K is about 4–8x and the pole gain
there is 0.034–0.037, the same as at 1x.

At 30K the ternary session's float polar 2x run scored 1.8396 against 1x's
1.8469, a gain of only 0.007 (0.033 at 3K), as Björck et al. predict: the 1x
30K pole comparison in text-pole-form is close to fairly tuned.
And Björck et al. 2024 says the best rate falls with more training, so these
3K numbers don't set the rate for a 30K run.

## What it changes

- Every queued 4L/32H experiment (QUEUE.md 1–5) compares against the 1x
  control. If the best rate is 2–4x, the control for those should move to
  it first, or their gains may partly be rate effects.
- The 30K headline (polar 1.8469 vs real 1.8841) is at 1x. A tuned-rate
  check at 30K is the follow-up this cannot replace, given Björck et al.
