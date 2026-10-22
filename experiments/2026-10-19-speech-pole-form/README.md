# Pole form on speech bytes: real decay vs polar vs cartesian complex pole

**Date.** 2026-10-19. **Status.** done (one seed). Backfilled 2026-10-22 from
the run records and `~/alpha/HANDOVER.md` §12.

## Question

Diakosmos's harmonic recurrence uses a real decay per channel. Does replacing it
with a complex pole (a damped rotation) lower held-out bits per byte on speech,
at equal size and updates?

## Prior art

- S4D ([Gu et al. 2022](https://arxiv.org/pdf/2206.11893)) — diagonal state
  spaces with complex eigenvalues; its initialization ablations are on
  sequential CIFAR, Speech Commands and BIDMC, not on a byte LM.
- Mamba's ablations found real and complex diagonals about equal on text and
  chose real for hardware reasons; one reading is that complex poles help
  continuous modalities (audio) and not discrete ones ([summary](https://sia.hackernoon.com/the-key-differences-between-real-and-complex-valued-state-space-models)).
  That is the open question here, on speech first.
- The audio-domain intuition (resonant filters have complex poles) is why the
  speech arm came before the text arm.

## Hypothesis and prediction

Complex poles let a channel resonate at a frequency, which a real decay cannot;
speech is made of resonances. Prediction (written in the session before the
run): a gain of a few percent in bpb for both complex forms, cartesian and
polar close to each other.

## Independent variable

The pole's form per channel: `real` (Diakosmos: c = e^-gamma, gamma > 0),
`polar` (c = e^-(gammaMin + e^rho) (cos w, sin w), the form Vidya kept), and
`cartesian` (c = a + b i with |c| projected below 1). Held fixed: 4 blocks,
width 448 (7 heads x 8 channels, 32 harmonics a group), 3.37–3.39 M
parameters, batch 16 x 2048 bytes, 3000 updates, Diakosmos's optimizer and
schedule, one seed. The complex forms double the recurrent state and the query
has an imaginary half (the 14,724 extra parameters).

## Code

`harness/zara_poles.py` (the speech harness; `diakosmos_oracle.py` with the
recurrence replaced), as of 2026-10-19; results in `<variant>/result.json`,
`<variant>/train.log`.

## Hardware and cost

RunPod community RTX 3070 (`roland-poles-3070`, $0.13/h), ~3.2 h in all
including two false starts (no rsync on the image, a missing import), ≈ $0.45.
Wall-clock per arm: real 50.6 min, polar 62.7 min, cartesian 63.5 min (the
complex arms run 25% slower in this PyTorch harness).

## Data

98 MB of speech bytes seen (3000 x 16 x 2048), held-out = all 4,688 windows
(9.6 M bytes). The speech corpus and its preparation are recorded in
`~/alpha/HANDOVER.md` §11 (not reproduced here).

## Results

| arm | params | held-out bpb | block-0 decays (ms) min / median / max | block-0 Hz range |
|---|---|---|---|---|
| real | 3,370,964 | 3.3964 | 0.05 / 0.18 / 0.69 | 0 |
| polar | 3,385,688 | **3.2933** | 0.04 / 0.31 / 1.18 | 25–6647 |
| cartesian | 3,385,688 | **3.2805** | 0.03 / 0.11 / 2.45 | 12–6110 |

Held-out subset at 500 / 1000 updates: real 3.69 / 3.44, polar 3.50 / 3.33.
Real's late slope (0.014 bpb per 500 updates on the subset) cannot close the
gap with its 25% time advantage.

Continuations (3 held-out prompts, 2 s, free-running): at T = 1.0 every arm
keeps speech loudness (RMS 0.03–0.05 against 0.04 in the truth) and log-mel
distance 1.4–3.0 against 5.9 for silence; at T = 0.8 all collapse to the
silence byte. Voiced-frame fraction at T = 1.0 (truth 0.60–0.72): real
0.09–0.30 with only 13% of frames active on one prompt; polar 0.14–0.45 with
92% active; cartesian 0.13–0.38 with 90% active.

Observed: every arm learned sub-millisecond decays in block 0 (1–20 samples),
i.e. the recurrence became a short resonant filter bank, and the complex arms
spread their frequencies over the band.

## Verdict

Prediction held: complex poles win 0.10–0.12 bpb (3%) at equal updates, the
two complex forms within 0.013 of each other. One seed; the size of the gap
is a hint, its sign is consistent with the text result that followed.

## What it changes

Carried the polar form into the text harness (the next experiment) and into
the Alpha text (Vidya.Pole). Suggested, not yet run: initial decays log-spaced
from 0.1 ms to 100 ms, since the learned decays sit at the short end of the
initial range.
