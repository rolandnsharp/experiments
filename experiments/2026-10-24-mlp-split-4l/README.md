# How much of the polar block's budget belongs in the MLP

**Record.** [EXP-0030](https://experiments.blah.dev/EXP-0030) on experiments.blah.dev (pre-registered
2026-10-24 10:33 UTC, before launch); protocol and predictions there; the numbers there and here match.

**Status.** concluded 2026-10-24: refuted on the optimum's location (20 harmonics, not 14 or 10).

At equal parameters (3,385,688 within 0.07%), the budget moves from recurrence harmonics (and the
width 14 x harmonics they set) into the SwiGLU MLP of EXP-0024:

| harmonics | width | MLP hidden | parameters |
|---|---|---|---|
| 32 | 448 | 0 (control, EXP-0005) | 3,385,688 |
| 26 | 364 | 256 | 3,383,584 |
| 20 | 280 | 600 (EXP-0024) | 3,383,272 |
| 14 | 196 | 1,144 | 3,385,648 |
| 10 | 140 | 1,794 | 3,386,112 |
| 6 | 84 | 3,210 | 3,385,456 |

Predicted (before any run): an interior optimum at h14 or h10; h6 worse than h20; h14 0–0.04 below h20.

## Results

| harmonics | width | MLP | seed 0 | seed 1 | mean | MAC/byte |
|---|---|---|---|---|---|---|
| 32 (control) | 448 | 0 | 2.0490 | 2.0409 | 2.0450 | 3.62 M |
| 26 | 364 | 256 | 1.9462 | 1.9456 | 1.9459 | 2.46 M |
| **20** | 280 | 600 | 1.9309 | 1.9303 | **1.9306** | 1.52 M |
| 14 | 196 | 1,144 | 1.9342 | 1.9393 | 1.9368 | 0.81 M |
| 10 | 140 | 1,794 | 1.9528 | 1.9477 | 1.9503 | 0.46 M |
| 6 | 84 | 3,210 | 1.9881 | 1.9850 | 1.9866 | 0.22 M |

Largest seed spread 0.005. 8 new runs, $1.02.

## Verdict

Prediction 1 (optimum at 14 or 10) missed: the minimum is at 20, between about 16 and 22. Prediction 2
(6 worse than 20) met: +0.056. Prediction 3 (14 below 20 by 0–0.04) missed: 14 is 0.006 above. The
quality choice for larger models is the h20 split; the compute choice is h14 (0.006 worse at 47% fewer
multiply-adds). Every split with an MLP beats the control by 0.058 or more, even at 6% of its compute.
