# Speech recognition, round 3: oscillator front vs mel on the MLP backbone

**Date.** 2026-10-23. **Status.** running (pod `roland-stt3-a40-1`, A40, Roland's account).
Prediction written before any result.

## Question

On a backbone that transcribes ([stt-gated-mlp](../2026-10-22-stt-gated-mlp/)), does the
learned bank of complex oscillators at the sample rate match the mel front end at equal
updates?

## Prior art

- SincNet ([Ravanelli & Bengio 2018](https://arxiv.org/abs/1808.00158)), LEAF
  ([Zeghidour et al. 2021](https://arxiv.org/abs/2101.08596)): learnable front ends roughly
  match mel. Their filters are FIR/Gabor convolutions; ours are recursive complex resonators
  with no FFT and no matrix, cost 4.1 M multiply-adds per second at 64 poles.
- Round 1 here ([stt-front-osc-vs-mel](../2026-10-21-stt-front-osc-vs-mel/)) and the earlier
  voice-command experiment: osc trails mel, by ~1,000 updates' worth early in training.

## Hypothesis and prediction

Written 2026-10-23 before the run produced a number:
- mel + MLP ends 30,000 updates at 0.38–0.45 WER on the whole of test-clean (greedy).
- osc + MLP ends 0.03–0.08 WER above mel (worse), the gap narrowing over training.
- The front's learned decays stay within ~0.3–50 ms (the init's range), centre frequencies
  stay spread over 50 Hz – 7 kHz.

What would change our mind: osc within 0.02 of mel or better means the oscillator front
earns its place (and is the one to ship, since it needs no FFT); osc worse by more than 0.08
means mel is the front to ship and the oscillator idea stays in the frame-rate stack only.

## Independent variable

The front end (osc, 64 poles; mel, 40 bands), as in round 1. Held fixed: gated MLP 448 after
every block, 4 blocks of width 448, ct poles at 50 Hz, 30,000 updates, batches of 240 s,
885.8 h of LibriSpeech training, seed 0, Diakosmos's optimizer. Parameters: osc 5.78 M,
mel 5.71 M.

## Code

`harness/stt/` as committed with this paper (`pod-stt2.sh` with `MLP=448 RESULTS=runs3`,
`retry2.sh`).

## Hardware and cost

RunPod secure-cloud A40 ($0.59/h), estimated 8–9 h ≈ $5.

## Data

As round 2; tests test-clean, test-other, Kokoro Zarathustra.

## Results

Pending.

## Decided vs measured

Decided: the parameter and multiply-add counts (osc + MLP 294 M, mel + MLP 287 M per second
of audio). Measured, one seed: every WER.

## Verdict

Pending.

## What it changes

Pending.
