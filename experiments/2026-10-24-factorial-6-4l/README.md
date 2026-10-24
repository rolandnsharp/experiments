# Six block changes in a 2^(6-1) factorial

**Record.** [EXP-0029](https://experiments.blah.dev/EXP-0029) on experiments.blah.dev (pre-registered
2026-10-24 10:25 UTC, before the launch); the protocol, predictions and analysis plan are there,
and the numbers there and here match.

**Status.** running: 27 legs on 6 community 3090/3080 pods (MARKS fx-1..fx-6), launched ~10:27 UTC;
the legs and their pods are in `design.txt`.

## Why a factorial

The one-at-a-time screens (EXP-0024 to EXP-0028) cannot see interactions: a change that helps only
with another present would fail its screen (Roland, 2026-10-24). A 2^(6-1) half fraction, I = -ABCDEF,
the 32 combinations with an odd number of the six on, measures all 6 main effects and all 15 two-way
interactions from one run each; five of its points are the screens' seed-0 runs.

Factors: A mlp (`--harmonics 20 --mlp 600` vs 32 harmonics), B select (`--select`), C phase
(`--phase`), D conv (`--conv 4`), E gate (`--gate 16`), F erase (`--erase`, the delta-rule erase
before each write, new: `harness/check_erase.py`).

## Results

(pending)
