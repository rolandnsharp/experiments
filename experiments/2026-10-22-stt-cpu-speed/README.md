# Speech recognition on the uConsole CPU: the recogniser's C port vs whisper.cpp

**Date.** 2026-10-22 (whisper, first port timings), 2026-10-23 (MLP variants).
**Status.** done for whisper.cpp and sherpa-onnx (2026-10-23); Vosk not timed. Our side is
timed on untrained weights (compute does not depend on them).

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

**Added 2026-10-23: the streaming baseline.** sherpa-onnx v1.13.8 (prebuilt aarch64 CPU
build, onnxruntime) with the streaming Zipformer transducer
`sherpa-onnx-streaming-zipformer-en-20M-2023-02-17`, int8, greedy search; `run-sherpa.sh`.
Mains power, every run pinned to one core with `taskset`, wall clock including model load
(`single-core-mains.txt`):

| one core | sherpa Zipformer 20M int8 | ours osc + MLP 448 | ours osc, no MLP | whisper tiny.en |
|---|---|---|---|---|
| A53, 10.4 s file | 1.10 | 0.70 | 0.49 | 4.49 |
| A72, 10.4 s file | 0.44 | 0.23 | 0.14 | 2.26 |
| A53, all 12 files (86.85 s) | **0.49** | 0.69 | — | — |
| A72, all 12 files (86.85 s) | **0.20** | 0.25 | — | — |

On one short file sherpa's load time (a 42 MB model) dominates and ours looks faster. Over
87 s of audio, with the load spread out, **sherpa is faster than our MLP model on both core
types**, 0.49 vs 0.69 on the A53 and 0.20 vs 0.25 on the A72. Its WER on the 12 utterances is
**0.086** (CER 0.058), against ours at 0.56 greedy (whole test-clean, trained checkpoint).

## Decided vs measured

Decided: multiply-adds per second of audio from the layout (osc 174 M, osc + MLP 294 M,
mel + MLP 287 M). Measured: every RTF and WER, on one machine, one run each, with the
uConsole's frequency governor left at schedutil.

## Verdict

Against the prediction: the A72 achieves 1.4 GMAC/s on this code, not 2–3, so RTF 0.12
rather than 0.06–0.09; the matrix-vector products are unvectorised scalar loops. On battery,
one slow core runs the working (MLP) recogniser just faster than real time (0.84) while
whisper tiny needs 2.25 on all four. That beats whisper, which is not a streaming model.
Against the honest baseline it does not hold: sherpa-onnx's 20 M streaming Zipformer, int8
on onnxruntime, is faster than our scalar-C MLP model on one core of either type and is at
8.6% WER on the same utterances where we are at 56% (greedy). Our architecture's arithmetic is
cheap (287 M MAC/s), but our implementation is unvectorised float32 and theirs is NEON int8.
There is no speed claim to make yet, and no accuracy claim.

## What it changes

A speed claim needs our implementation to match theirs in kind: NEON-vectorised (and
possibly int8) matrix-vector products, which is where nearly all our time goes. Even then the
accuracy gap (8.6% vs 56% greedy) is the larger problem; the speed comparison only matters for
a model that transcribes usefully. Vosk is not timed.
