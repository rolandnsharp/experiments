# Five block changes in a 2^(5-1) factorial (planned as six)

**Record.** [EXP-0029](https://experiments.blah.dev/EXP-0029) on experiments.blah.dev (pre-registered
2026-10-24 10:25 UTC, before the launch); the protocol, predictions and analysis plan are there,
and the numbers there and here match.

**Status.** running: 11 legs on 3 community 3090/3080 pods (MARKS fx5-1..fx5-3), launched ~10:40 UTC;
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

(pending)
