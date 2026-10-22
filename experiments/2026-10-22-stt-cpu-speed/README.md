# Speech recognition on the uConsole CPU: the recogniser's C port vs whisper.cpp

**Date.** 2026-10-22 (whisper, first port timings), 2026-10-23 (MLP variants).
**Status.** partial: our side is timed on untrained weights; the streaming baselines that
matter (sherpa-onnx, Vosk) are not timed yet.

## Question

How fast does this architecture transcribe on the uConsole's CPU (ClockworkPi CM4-class
board: 4 × Cortex-A53 always on, 2 × Cortex-A72 only on mains power), compared with
whisper.cpp on the same machine?

## Prior art

- Whisper ([Radford et al. 2022](https://arxiv.org/abs/2212.04356)) encodes 30 s windows
  with attention; whisper.cpp is the standard CPU port. It is built for accuracy, not
  streaming, so it is an easy speed baseline.
- Moonshine ([Jeffries et al. 2024](https://arxiv.org/abs/2410.15608)) cuts whisper's cost
  on short audio with variable-length encoding, for edge devices.
- Streaming Zipformer transducers in k2 / sherpa-onnx
  ([Yao et al. 2023](https://arxiv.org/abs/2310.11230)) and Kaldi-based Vosk already run
  faster than real time on Raspberry Pi-class boards with far better accuracy than ours.
  **They are the real baseline for a speed claim**, and are not measured here yet.

## Hypothesis and prediction

Written in the session before the port existed: at 174 M multiply-adds per second of audio,
one A72 at 2–3 GMAC/s gives a real-time factor of 0.06–0.09.

## Independent variable

The recogniser (whisper.cpp tiny.en / base.en vs our C port), and the power state
(battery: the A72s are off).

## Code

whisper.cpp at `~/src/whisper.cpp` (ggml-tiny.en, ggml-base.en), `run-whisper.sh`,
`wer.py`, `refs.txt` in this folder. Our side: `harness/stt/zara_stt.c` (single file, plain
C, libm only, single-threaded, streaming: the oscillator bank or the STFT advances sample by
sample, the blocks frame by frame, all state in fixed-size arrays), exported by
`harness/stt/export_c.py`. Verified against the PyTorch forward: max |Δ log-prob| 4e-6 (osc),
2e-5 (osc + MLP), 1.3e-3 (mel, float32 FFT), identical transcripts. Built with
`cc -O3 -ffast-math` (`-mcpu=cortex-a72` made no difference).

## Hardware and cost

The uConsole itself and the lab (Haswell VPS core) for comparison. No cloud cost.

## Data

Whisper: the first 12 test-clean utterances (86.85 s). Ours: utterance 1089-134686-0000
(10.4 s). Our compute does not depend on the weights, so untrained full-size models give
the right timing; they give no accuracy.

## Results

Real-time factor = compute seconds per second of audio (below 1 keeps up with speech).

| recogniser | threads | power | RTF | WER (12 utt) |
|---|---|---|---|---|
| whisper.cpp tiny.en | 4 A53 | battery | 2.25 | 0.050 |
| whisper.cpp tiny.en | 4 A53 + 2 A72 | mains | 0.84 | 0.050 |
| whisper.cpp base.en | 4 A53 + 2 A72 | mains | 1.98 | 0.041 |

| ours (C port, one core) | params | core | power | RTF |
|---|---|---|---|---|
| osc, no MLP | 3.37 M | A72 1.8 GHz | mains | 0.12 |
| osc, no MLP | 3.37 M | A53 | mains | 0.40 |
| osc, no MLP | 3.37 M | A53 | battery | 0.50 |
| osc + MLP 448 | 5.78 M | A53 | battery | 0.84 |
| mel + MLP 448 | 5.71 M | A53 | battery | 0.83 |
| osc, no MLP | 3.37 M | lab Haswell | — | 0.067 |
| mel, no MLP | 3.30 M | lab Haswell | — | 0.097 |

The MLP variants on an A72 (mains) are not timed yet.

## Decided vs measured

Decided: multiply-adds per second of audio from the layout (osc 174 M, osc + MLP 294 M,
mel + MLP 287 M). Measured: every RTF and WER, on one machine, one run each, with the
uConsole's frequency governor left at schedutil.

## Verdict

Against the prediction: the A72 achieves 1.4 GMAC/s on this code, not 2–3, so RTF 0.12
rather than 0.06–0.09; the matrix-vector products are unvectorised scalar loops. On battery,
one slow core runs the working (MLP) recogniser just faster than real time (0.84) while
whisper tiny needs 2.25 on all four. That beats whisper on speed and memory by a wide margin
and says nothing yet against the streaming recognisers that are the honest comparison; and
whisper is at 5% WER where ours is at 56% greedy / 23% with LM.

## What it changes

Time sherpa-onnx streaming Zipformer and Vosk on the same utterances; NEON-vectorise the
matrix-vector products (the obvious 2–4x); time the MLP variants on the A72.
