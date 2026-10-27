# Speech recognition, round 4: train longer, with augmentation

**Date.** 2026-10-23. **Status.** concluded 2026-10-24: **prediction failed**, round 4 is slightly worse than round 3 on clean speech. Ran 2026-10-23 08:25 UTC to 2026-10-24 01:43 UTC (A40, $0.59/h).
Prediction written before any run.

Two false starts (data preparation too slow, then out of memory; ≈ $0.70) preceded the run reported here; augmentation batches are prepared by 6 worker processes (`--workers 6`), 1.07 s/update.

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

## Data

LibriSpeech train-clean-100 + train-clean-360 + train-other-500 (886 h), as round 3. Scored on
all of test-clean (2,620) and test-other (2,939), the Kokoro synthetic set (322, never trained
on), and, for the language-model weights, the first 300 utterances of dev-clean.

## Results

Python harness, greedy CTC, whole sets, final parameters (`params.pt`) of each round:

| | round 3 (30k updates) | round 4 (+60k, augmented) | predicted for round 4 |
|---|---|---|---|
| test-clean | 0.386 | **0.399** | 0.25–0.30 |
| test-other | 0.621 | **0.608** | 0.45–0.52 |
| Kokoro | 0.370 | **0.400** | 0.28–0.33 |

C port, int8, beam search with the LibriSpeech 3-gram, alpha and beta chosen on dev-clean 300
(same protocol, `port/tune-model.sh`, round 3 in `tune-dev-r3osc.txt`):

| | round 3 | round 4 |
|---|---|---|
| dev-clean 300, greedy | 0.415 | 0.434 |
| chosen on dev | alpha 0.5, beta 1.0 (dev 0.221) | alpha 0.5, beta 2.0 (dev 0.230) |
| test-clean 2620, greedy | 0.390 | 0.399 |
| test-clean 2620, beam 32 + 3-gram | 0.203 | **0.215** |
| test-clean 2620, beam 64 + 3-gram | 0.194 | **0.207** |

The prediction named the 200-utterance subset with the LM; the whole test set under the
dev-tuned protocol replaced it (the subset numbers were found optimistic in
`2026-10-23-stt-c-port-lm-int8`), and both rounds are scored the same way here.

Training curve, test-clean subset (200 utt), greedy: round 3 ended at 0.334. Round 4 was 0.479
after its first 1,000 updates, stayed at 0.46–0.48 until update 25,000, then fell with the cosine
schedule to 0.356 at the end, still falling (0.363 → 0.356 over the last 5,000). The Kokoro subset
followed the same shape (0.367 → 0.474 → 0.396).

## Decided vs measured

Measured: the numbers above. Decided, not measured: that the jump at the start comes from the
fresh warmup to the peak learning rate together with the switch to augmented data (the run cannot
separate them, as the independent-variable section says).

## Verdict

Prediction failed on every count. The paper's own branch applies: worse than 0.30 with test-other
barely moving (0.621 → 0.608) means augmentation too strong for this model size, or, read
together with the curve, the restart threw away most of what round 3 had learned and 60,000
augmented updates only won it back. At 5.8 M parameters the model seems to spend its capacity on
the harder augmented data; clean speech pays for it and noisy speech gains little.

## What it changes

- The release stays on round 3 (`osc-mlp448-r3.bin`).
- Next lever is capacity, not schedule: a wider or deeper model trained from scratch, with
  augmentation as a separate, single variable (a no-augmentation control at the same budget).
- If round 3 is ever resumed, resume with the learning rate where it ended, not a new warmup.

## Hardware and cost (actual)

A40, 17.3 h ≈ $10.20, plus the two false starts ≈ $0.70.
