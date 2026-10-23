# Speech recognition, round 4: train longer, with augmentation

**Date.** 2026-10-23. **Status.** running since 2026-10-23 08:25 UTC (third start; pod `roland-stt4-a40-1`, A40, $0.59/h).
Prediction written before any run.

Run note: the first start ran at 2.05 s/update because the augmentation (numpy, one CPU core) starved
the GPU; it was stopped after ~150 updates and restarted from the same checkpoint with batches
prepared by 6 worker processes (`--workers 6`): 1.07 s/update, about 18 h for 60,000 updates.

Second failure: that run was killed by the kernel (exit 137, out of memory) at update ~900. Cause:
`Pool.imap` pulls jobs eagerly from an infinite generator, so the workers kept preparing 15 MB
batches faster than the GPU consumed them until the pod's 48 GB was full. Fix: a semaphore bounds
the batches prepared ahead to 2 x workers (verified on the lab: workers wait, memory flat). The
driver treated the death as the end of the run and stopped the pod; the run was relaunched from
the same round-3 checkpoint on a new A40 (`x3qd3mmimou0wf`) at 08:25 UTC. Cost of both false
starts: about $0.70 and 1.5 h.

## Question

The round-3 oscillator recogniser (gated MLP, 30,000 updates, 2.3 passes over 886 h) was still
improving when it stopped (subset WER 0.355 → 0.334 over the last 5,000 updates), and it is
trained only on clean read audiobooks while it is used live on a headset in a room. How much do
more updates and augmentation buy?

## Prior art

- SpecAugment ([Park et al. 2019](https://arxiv.org/abs/1904.08779)): time and frequency
  masking gave large relative gains on LibriSpeech for attention and CTC models.
- Speed perturbation ([Ko et al. 2015](https://www.danielpovey.com/files/2015_interspeech_augmentation.pdf))
  and simulated reverberation ([Ko et al. 2017](https://www.danielpovey.com/files/2017_icassp_reverberation.pdf))
  are standard in Kaldi recipes for robustness to speaking rate and rooms.
- What is open: the size of the gain for this model, which is far from converged; part of any
  gain is just more updates.

## Hypothesis and prediction

Resuming the round-3 osc checkpoint for 60,000 more updates with `--augment`:
- test-clean (whole), greedy: 0.386 → **0.25–0.30**; test-other 0.621 → **0.45–0.52** (augmentation
  helps the noisier set more).
- C port with the 3-gram on the 200-utterance subset: 0.166 → **0.11–0.13**.
- Kokoro (synthetic voice, never trained on): 0.370 → 0.28–0.33.

What would change our mind: test-clean better than 0.25 means the model was badly undertrained
and a longer schedule is the main lever; worse than 0.30 with test-other barely moving means the
augmentation is too strong for this model size (rerun without it to separate the two).

## Independent variable

Two things move together, deliberately, because the budget allows one run: the number of updates
(30,000 → 90,000 in total, a fresh warmup and cosine schedule for the last 60,000) and augmentation
(speed 0.9/1.0/1.1; with probability 0.5 each: synthetic room RT60 0.1–0.6 s, spectral tilt,
coloured noise 5–35 dB SNR; two time masks of up to 10 frames and two channel masks of up to 10%
of the features). The paper will not attribute the gain to either alone. Held fixed: the
architecture (osc, 64 poles, 4 blocks, width 448, MLP 448), the data, batch 240 s, the optimizer.

## Code

`harness/stt/zara_stt.py` (`--augment`, `augment_audio`, `mask_features`, `--init`),
`pod-stt2.sh` (`INIT_FILE`, `AUGMENT`), `pod-run-all.sh`.

## Hardware and cost

Planned: one A40 (~$0.59/h) about 16 h ≈ $10, or a 3090 ($0.22/h) if available.

## Data, Results, Decided vs measured, Verdict, What it changes

Pending.
