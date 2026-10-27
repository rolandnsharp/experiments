# Five block changes in a 2^(5-1) factorial (planned as six)

**Record.** [EXP-0029](https://experiments.blah.dev/EXP-0029) on experiments.blah.dev (pre-registered
2026-10-24 10:25 UTC, before the launch); the protocol, predictions and analysis plan are there,
and the numbers there and here match.

**Status.** all 16 points done (2026-10-24); confirmation of the fitted best running (2026-10-26). 11 legs ran on 3 community 3090/3080 pods (MARKS fx5-1..fx5-3), launched ~10:40 UTC;
the legs and their pods are in `design.txt`.

**Deviation (10:39 UTC, before any run).** Planned as a 2^(6-1) with the delta-rule erase as a sixth
factor (27 legs, 6 pods, launched ~10:27); none of those pods had rented when the erase was dropped to
save compute (Roland): least expected gain at 8 key dimensions a head, the most expensive runs, and
better tested later on the winning combination at a larger size. Now a 2^(5-1), I = -ABCDE, resolution
V: every main effect and two-way interaction estimable from 16 points, 5 of them the screens' seed-0 runs.

## Why a factorial

The one-at-a-time screens (EXP-0024 to EXP-0028) cannot see interactions: a change that helps only
with another present would fail its screen (Roland, 2026-10-24). A 2^(6-1) half fraction, I = -ABCDEF,
the 32 combinations with an odd number of the six on, measures all 6 main effects and all 15 two-way
interactions from one run each; five of its points are the screens' seed-0 runs.

Factors: A mlp (`--harmonics 20 --mlp 600` vs 32 harmonics), B select (`--select`), C phase
(`--phase`), D conv (`--conv 4`), E gate (`--gate 16`). (The dropped F erase, `--erase`, is in the
harness with its checks, `harness/check_erase.py`.)

## Results

All 16 design points (seed 0; the five singles are the screens' runs, the select single its capped
rerun), held-out bpb over all 18,538,212 scored bytes:

| point | bpb | point | bpb | point | bpb |
|---|---|---|---|---|---|
| m | 1.9309 | msp | 1.9310 | spc | 2.0053 |
| s | 2.0036 | msc | 1.9142 | spg | 1.9832 |
| p | 2.0307 | msg | **1.9124** | scg | 1.9452 |
| c | 2.0072 | mpc | 1.9378 | pcg | 1.9702 |
| g | 1.9771 | mpg | 1.9356 | mspcg | 1.9333 |
| | | mcg | 1.9161 | | |

Fit (`analyse.py`, the pre-registered model): effects 2b, Lenth PSE 0.0047, seed sd 0.0034 from 6
replicate pairs (effect se 0.0017):

| effect | 2b | | effect | 2b |
|---|---|---|---|---|
| mlp | **-0.0639** | | mlp x gate | **+0.0193** |
| gate | **-0.0234** | | mlp x conv | +0.0073 |
| select | **-0.0097** | | select x phase | +0.0043 |
| conv | -0.0094 | | select x gate | +0.0035 |
| phase | **+0.0150** | | others | under 0.003 |

Fitted best of all 32: mlp+select+conv+gate 1.9092; then mlp+select+gate 1.9124 (measured), mlp+select+conv
1.9142 (measured), mlp+select 1.9144, mlp+conv+gate 1.9161 (measured). Control fitted 2.0247, measured 2.0450.

Readings (the verdict waits for the confirmation runs): the MLP dominates; the gate helps but overlaps with
the MLP (the largest interaction, +0.019, beyond the predicted 0.015); the input-dependent rotation *hurts*
once combined (+0.015); damping and convolution add about 0.01 each. Confirmation of mlp+select+conv+gate on
seeds 1 and 2 launched 2026-10-26 (MARKS fxbest-1, fxbest-2).
