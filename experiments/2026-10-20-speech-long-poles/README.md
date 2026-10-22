# Long timescales on speech bytes: the continuous-time pole (decays to 4 s)

**Date.** 2026-10-20. **Status.** done (one seed). Written up 2026-10-23 from the run records
and `~/alpha/HANDOVER.md` §13.

## Question

In [speech-pole-form](../2026-10-19-speech-pole-form/) every variant learned sub-millisecond
decays and nothing longer than the polar form's 125 ms cap. Given poles that start at up to
2 s and a 16,384-byte (1 s) window, does a speech byte model use long timescales, or does
training pull them back to short resonances?

## Prior art

- S4D ([Gu et al. 2022](https://arxiv.org/abs/2206.11893)) and LRU
  ([Orvieto et al. 2023](https://arxiv.org/abs/2303.06349)) initialise eigenvalues over a
  range of timescales and show the initialisation matters for long-range tasks; neither
  reports what a byte-level speech model does with long poles.
- The speech-pole-form run in this repo: all learned decays under ~20 ms.

## Hypothesis and prediction

Backfilled honestly: the prediction stated in the session before the run was qualitative.
No bpb gain was expected from a timescale change alone (the question was whether long poles
survive, which informs the recogniser's frame-rate initialisation). No numeric bpb prediction
was written, which breaks rule 2.

## Independent variable

The pole parameterisation and its initialisation: `ct` (continuous time),
r = exp(-dt (1/4 s + 1/tau)), theta = 2 pi dt f, tau log-spaced 0.1 ms to 2 s,
harmonic-major so every head spans every timescale, f = 2 / (pi tau) at the start
(0.32 Hz to 6.4 kHz). Also moved, so this is not a clean single-variable comparison: the
window (16,384 bytes instead of 2,048) and the batch (4 instead of 16), keeping bytes per
update equal (65,536). Held fixed: 4 blocks, width 448, 3.39 M parameters, 3,000 updates,
Diakosmos's optimizer and schedule, one seed.

## Code

`harness/stt/zara_poles.py` (the `ct` variant, the log-depth `complex_scan`: 64-step chunks
in closed form, chunk-end states scanned recursively; 6x faster than the per-chunk loop).
Verified on the lab CPU: scan equal to the position loop to 3.4e-14, finite-difference
gradients for the ct parameters to 2.2e-4.

## Hardware and cost

RunPod community RTX 3070 (Ajax's account, `roland-poles-3070`, $0.13/h), 71 min of
training (4,241 s), about 2 h of pod time with scoring and generation, ≈ $0.27.

## Data

As speech-pole-form: Thus Spake Zarathustra read by Kokoro TTS (one voice), 16 kHz mu-law,
held-out = the last 600 s (9.6 M scored bytes). 196 MB of training bytes seen.

## Results

Held-out bpb on all 9.6 M scored bytes, every arm scored at window 16,384
(`rescore.jsonl`; the window-2,048 numbers of the earlier arms did not move):

| arm | window trained | bpb |
|---|---|---|
| real | 2,048 | 3.390 |
| polar | 2,048 | 3.285 |
| cartesian | 2,048 | 3.271 |
| ct | 16,384 | 3.280 |

Learned decays per block (min / median / max), initialised 0.1 / 14 / 1,330 ms:

| block | decay ms | Hz |
|---|---|---|
| 0 | 0.025 / 6.2 / 1,010 | 0.37 – 5,000 |
| 1 | 0.065 / 6.5 / 1,140 | 0.33 – 5,910 |
| 2 | 0.069 / 6.6 / 1,150 | 0.31 – 6,110 |
| 3 | 0.051 / 5.6 / 1,170 | 0.35 – 6,340 |

Free-running continuations from three 1 s held-out seeds (log-mel distance to the truth;
silence scores 5.88): temperature 1.0: 2.49, 23% silent frames; temperature 0.8: 3.45,
72% silent. Polar and cartesian scored 2.12 and 2.21 at temperature 1.0. The audio is
speech-shaped, not intelligible. Checkpoint: lab `~/zara/poles2/ct/params.pt`,
md5 9f9a728a28db2cd6fd0b2bb0004933e8.

## Decided vs measured

Decided (could be laws in Alpha, not yet written): |c| < 1 for every parameter value of the
ct form, since log r ≤ -dt/4 s < 0; the scan equals the position loop (checked numerically
here, not proved). Measured, one seed: every bpb and continuation number.

## Verdict

Long poles survive training: every block kept a group at 1.0 to 1.2 s. The middle group moved
shorter (14 → 6 ms) and the fast group to 0.02 – 0.07 ms. Bits per byte are level with the
other complex forms (within 0.015 bpb), so long memory neither helps nor hurts next-byte
prediction at this size. One seed; the window and batch also changed.

## What it changes

The recogniser's frame-rate blocks start with decays from 20 ms to 4 s (ct form, at 50 Hz).
The speech continuation line stops here: intelligible continuation needs scale we do not
have; see [stt-front-osc-vs-mel](../2026-10-21-stt-front-osc-vs-mel/).
