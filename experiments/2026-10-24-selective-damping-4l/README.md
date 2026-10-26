# Input-dependent damping of the polar pole

**Record.** [EXP-0025](https://experiments.blah.dev/EXP-0025) on experiments.blah.dev; the numbers there and here match. This protocol was committed to git before launch while the site was unreachable; the site record was created from it and pre-registered at 10:02–10:04 UTC, and its methodology states which runs had started by then.

**Date.** 2026-10-24. **Status.** concluded 2026-10-26: supported (gain just above the predicted range, at 2.2x the step time).

## Question

The polar pole c = exp(-(gammaMin + e^rho)) e^(i omega) has a fixed magnitude per harmonic.
Does letting the input set the damping, c_t = exp(-(gammaMin + e^(rho + W u_t))) e^(i omega),
with the phase kept fixed, lower held-out bpb?

## Prior art

- LRU ([Orvieto et al. 2023](https://arxiv.org/abs/2303.06349)): the fixed polar
  parametrization we use.
- RG-LRU in Hawk/Griffin ([De et al. 2024](https://arxiv.org/abs/2402.19427)): an LRU whose
  decay is gated by the input; GLA ([Yang et al. 2023](https://arxiv.org/abs/2312.06635)) and
  Mamba ([Gu and Dao 2023](https://arxiv.org/abs/2312.00752)): data-dependent decay improves
  linear recurrences in language modelling. Borrowed: input-dependent decay. Ours: keeping
  the phase fixed and complex (the rotation stays a learned constant; only the damping
  moves), in the exp-exp form so the modulus stays below 1 for every input, at our harness's
  recipe.

## Hypothesis and predictions

Written before any run.

1. The selective arm scores **0.01–0.04 bpb below** the control, as the mean over seeds 0
   and 1, and below it on each seed. A mean gain under 0.01, or the selective arm worse on
   either seed, refutes it.
2. Its step time is within 1.5x the control's.

## Variables

- Independent: pole damping, fixed (control, polar 4L/32H, 3,385,688 parameters) vs input-
  dependent (`--select`, 3,486,040 parameters, +3.0%: a (H*L, D) projection per block,
  `blocks.i.temporal.attn.pole_decay_select`, initialized to zero so the arm starts exactly
  as the control, trained with Adam).
- The +3% parameters are a known confound, left in: at text-pole-form's scaling a 3%
  parameter change is worth well under 0.01 bpb, below the refutation threshold.
- Held fixed: as the gated-MLP experiment (4L, 3,000 updates, lr 1x, batch 16, 2 shards,
  seeds 0 and 1, the full held-out set).

## Code

`--select` in `harness/zara_poles.py`: `recurrence_selective` uses per-position cumulative
log-magnitudes inside each chunk. Checked on the lab's CPU: equals a position-by-position
loop to 3.5e-14; with constant damping equals `recurrence()` to 1.8e-14; at initialization
the parameters and loss equal the control's (difference 0); chunked equals the sampler to
1.7e-5; with the flag off the harness is bit for bit the previous one.

## Runs and data

As the gated-MLP experiment; one community RTX 3090/3080 on Roland's account:
`LEGS="polar:dense:3000:0 polar:dense:3000:1" LAYERS=4 HARMONICS=32 BATCH=16 SHARDS=2
EXTRA="--select" TAG=sel`. Control from text-pole-form: 2.0490 (seed 0), 2.0409 (seed 1).

## Analysis

As the gated-MLP experiment. A pass earns runs at more updates and larger sizes.

## Results

| seed | control | input-dependent damping (capped) | gain |
|---|---|---|---|
| 0 | 2.0490 | 2.0036 | 0.045 |
| 1 | 2.0409 | 2.0045 | 0.036 |

Mean gain **0.041**, 1.64 s an update against 0.75 s. The first seed-0 run, before the cap at 4
(commit 7041d8a), overflowed float32 and went to NaN at update ~390 (recorded as failed; a first
rental whose container never started is recorded as abandoned).

## Verdict

Prediction 1 missed on the high side by 0.001 (0.041 against 0.01–0.04); no loss on either seed.
Prediction 2 missed: 2.2x the step time (the per-position decay products in the PyTorch chunked path).
In the factorial (EXP-0029) its effect averaged over the other changes is -0.010.
