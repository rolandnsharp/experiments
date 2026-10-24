# How much of the polar block's budget belongs in the MLP

**Record.** [EXP-0030](https://experiments.blah.dev/EXP-0030) on experiments.blah.dev (pre-registered
2026-10-24 10:33 UTC, before launch); protocol and predictions there; the numbers there and here match.

**Status.** running: 8 legs on 2 community 3090/3080 pods (MARKS ms-0, ms-1), launched ~10:34 UTC.

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

(pending)
