# Speech recognition, round 3: oscillator front vs mel on the MLP backbone

**Date.** 2026-10-23. **Status.** osc arm done, mel arm running (pod `roland-stt3-a40-1`, A40, Roland's account).
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

**osc arm (done 2026-10-23 05:38 UTC; mel arm running).** 30,000 updates, 7 h 52 min on the
A40 (0.94 s/update, the GPU about 44% busy: the sample-rate oscillator scan is the bottleneck).
Checkpoint lab `~/zara/stt/runs3/runs/osc/params.pt`, md5 c7daf7a804d653f9d228b83782b06dcb.

Subset curve (first 200 utterances), WER test-clean / Kokoro:

| step | 5K | 10K | 15K | 20K | 25K | 30K |
|---|---|---|---|---|---|---|
| test-clean | 0.664 | 0.513 | 0.434 | 0.380 | 0.355 | 0.334 |
| Kokoro | 0.623 | 0.499 | 0.437 | 0.398 | 0.379 | 0.367 |

Final, whole test sets, greedy (WER / CER): test-clean **0.386 / 0.120**, test-other
0.621 / 0.253, Kokoro 0.370 / 0.112.

Learned front: centre frequencies 30.6 Hz – 9.34 kHz, decays 0.30 ms / 7.5 ms (median) /
1,000 ms. The 1,000 ms is the cap (TAU_FRONT_MAX = 1 s): some front poles ran to the longest
decay allowed. 9.34 kHz is above the 8 kHz Nyquist frequency, so that pole rings as its alias
(6.66 kHz); the parameterisation does not prevent this.

Through the C port (`c-port-eval200.txt`, first 200 test-clean utterances, 16-bit audio,
chunks of 16 frames): greedy 0.334 (float32) / 0.339 (int8); beam 32 + LibriSpeech 3-gram
0.165 / **0.166**. On the uConsole, int8 + LM, one core, 12 utterances (86.85 s) including
loading the model and the LM: real-time factor 0.121 on an A72, 0.251 on an A53, WER 0.118
(sherpa-onnx's 20 M Zipformer on the same 12: 0.199 / 0.500, WER 0.086 greedy, no LM).

## Decided vs measured

Decided: the parameter and multiply-add counts (osc + MLP 294 M, mel + MLP 287 M per second
of audio). Measured, one seed: every WER.

## Verdict

Pending the mel arm. Already against the prediction: the front's decays did not stay within
0.3–50 ms; the longest went to the 1 s cap.

## What it changes

Pending.
