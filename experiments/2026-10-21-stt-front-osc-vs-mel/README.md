# Speech recognition, round 1: oscillator front end vs mel, on the Diakosmos block

**Date.** 2026-10-20. **Status.** done, inconclusive (osc arm crashed at step ~3,160).
Written up 2026-10-23.

## Question

For a CTC character recogniser whose frame-rate stack is Diakosmos's block with ct poles,
does a learned bank of damped complex oscillators at the sample rate (no FFT, no matrix)
match a 40-band log-mel front end?

## Prior art

- SincNet ([Ravanelli & Bengio 2018](https://arxiv.org/abs/1808.00158)) and LEAF
  ([Zeghidour et al. 2021](https://arxiv.org/abs/2101.08596)) learn the front-end filters from
  raw audio; LEAF reports roughly mel-level results across tasks, with gains on some. Neither
  uses recursive complex resonators feeding a recurrent stack, nor reports cost per second.
- Roland's earlier voice-command experiment on the lab (`~/zara/kws.py`, `kws_learn.py`):
  mel 86% vs fixed oscillator bank 73% on unseen voices, learned bank 78%.

## Hypothesis and prediction

No numeric prediction was written before this run, which breaks rule 2. The expectation
from the voice-command experiment was that osc trails mel early.

## Independent variable

The front end. `osc`: 64 poles z_t = c z_{t-1} + x_t, c = exp(-dt/tau + 2 pi i f dt),
f log-spaced 50 Hz – 7 kHz, Q = 8 at the start, tau and f learned; every 20 ms the
features are Re z, Im z (times 1 - |c|) and log mean |z|² over the frame (192 numbers).
`mel`: 40 log mel energies, 25 ms Hann, 512-point FFT, hop 20 ms. Held fixed: layer norm →
linear → 4 Diakosmos blocks (width 448, ct poles at 50 Hz, decays 20 ms – 2.7 s) → layer
norm → linear to 29 classes (blank, apostrophe, space, A–Z); CTC; batches of 120 s of audio;
6,000 updates; Diakosmos's optimizer; one seed. Parameters: osc 3.37 M, mel 3.30 M.

## Code

`harness/stt/zara_stt.py` as of 2026-10-20 (before `--mlp`, periodic saves and multi-corpus
training), `pack_libri.py`.

## Hardware and cost

RunPod community RTX 3070 (Ajax's account, `roland-stt-3070`), 0.6 s/update; ≈ 3 h of pod
time including a 17 min LibriSpeech download and a stalled driver, under $0.50.

## Data

LibriSpeech train-clean-100 (92.5 h, utterances ≤ 16 s), test-clean (all 2,620 utterances for
the final score, the first 200 for the curve). 0.72 M seconds of audio seen ≈ 2.2 passes.

## Results

| arm | step 3,000 subset WER / CER | final test-clean WER / CER |
|---|---|---|
| mel | 0.874 / 0.364 | 0.860 / 0.369 (6,000 updates) |
| osc | 0.942 / 0.440 | crashed at step ~3,160 (CUDA out of memory: allocator fragmentation, 3.15 GiB reserved but unusable on the 8 GB card) |

The harness saved parameters only at the end, so the osc run left only its log
(`osc/train.log`). Compute per second of audio: osc 174 M multiply-adds, mel 167 M
(the frame-rate blocks are 165 M of either).

## Decided vs measured

Decided: the multiply-add counts follow from the layout (`macs_per_second`). Measured, one
seed: every WER.

## Verdict

Inconclusive on the front: osc trailed mel by about 1,000 updates' worth of progress at the
point it crashed, as in the voice-command experiment. The larger finding is that mel itself
plateaued near 85% WER, so the backbone, not the front, limits this model.

## What it changes

Fix the harness (save every evaluation, allocator setting, multi-corpus data), train on all
960 h, and find out why the backbone plateaus before comparing fronts again:
[stt-gated-mlp](../2026-10-22-stt-gated-mlp/).
