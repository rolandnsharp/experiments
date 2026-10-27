# Speech recognition, round 2: does the Diakosmos block need a pointwise nonlinearity?

**Date.** 2026-10-22. **Status.** done (one seed; the two arms differ in schedule length,
see below).

## Question

The mel recogniser plateaued near 85% WER on 100 h and looked like doing the same on 960 h.
The Diakosmos block has no pointwise nonlinearity: a linear recurrence over products of
linear projections, a rotation, layer norms and gates. Is that the bottleneck?

## Prior art

- Diagonal state-space and linear-recurrent models interleave the recurrence with a gated
  MLP or GLU: S4/S4D ([Gu et al. 2022](https://arxiv.org/abs/2111.00396)), Mamba
  ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752)), LRU
  ([Orvieto et al. 2023](https://arxiv.org/abs/2303.06349)). Their speech-recognition uses
  (e.g. in ESPnet) keep that MLP. The open question was only whether its absence explains
  this plateau, and by how much.
- Diakosmos was designed for next-byte prediction, where the plateau did not show.

## Hypothesis and prediction

Written in the session after the step-5,000 evaluation of the plain arm and before the MLP
arm ran: the plain arm ends 30k updates at 65–75% WER on the test-clean subset; a gated MLP
is the bottleneck if it moves the step-5,000 subset WER from 0.817 to the low 0.70s or better.
If it does not, the next suspect is look-ahead (the model is causal).

## Independent variable

A gated MLP after every block: z += sigmoid(g) · W2 (silu(W1 u) ⊙ W3 u) + b, u = ln(z),
hidden 448, one learned scalar gate per block (W1, W3, W2 are row-normed by the oracle's
"rotational" update, so the gate carries the branch's scale). +2.41 M parameters
(3.30 M → 5.71 M), +120 M multiply-adds per second of audio (167 M → 287 M).

**Also moved:** the update budget, so the schedules differ. The plain arm ran 30,000
updates; the MLP arm 10,000 (the GPU time freed by skipping the osc run on the plain
backbone). The learning-rate schedule (warmup 200, cosine to 10%) is tied to the budget,
so at step 10,000 the MLP arm has finished its decay and the plain arm is a third of the way
through. The equal-step comparison below favours the MLP arm by that much; the final
comparison (MLP at 10k vs plain at 30k) favours the plain arm.

Held fixed: mel front, 4 blocks of width 448 with ct poles at 50 Hz, CTC over 29 classes,
batches of 240 s of audio, Diakosmos's optimizer, seed 0.

## Code

`harness/stt/zara_stt.py` (`--mlp`, `mlp_step`), `pod-run-all.sh`, `pod-stt2.sh`,
`retry2.sh`, `pod-probe.sh`. The corpus is memory-mapped (`np.memmap`).

## Hardware and cost

RunPod secure-cloud A40 48 GB (Roland's account, $0.59/h; no community 3090/4090 was
available that day): plain arm 2 h 32 min, MLP arm 56 min, data download and packing 1 h.
≈ $3.50 in all, including two failed attempts.

## Data

LibriSpeech train-clean-100 + train-clean-360 + train-other-500 (885.8 h, utterances
≤ 16 s, 264,723 utterances), packed to 16 kHz mu-law bytes on the pod. Plain arm: 7.2 M s
of audio seen ≈ 2.3 passes; MLP arm 2.4 M s ≈ 0.75 passes. Tests: test-clean (2,620),
test-other (2,939), and Kokoro-read Zarathustra paragraphs (322 utterances, 1.39 h, one
synthetic voice, never in training). Curves on the first 200 utterances of each.

## Results

Subset WER at equal steps (test-clean / test-other / Kokoro):

| step | plain | MLP |
|---|---|---|
| 1,000 | 0.918 / 0.950 / 0.925 | 0.905 / 0.944 / 0.905 |
| 5,000 | 0.817 / 0.903 / 0.774 | 0.626 / 0.802 / 0.582 |
| 10,000 | 0.779 / 0.884 / 0.738 | 0.513 / 0.725 / 0.488 |
| 20,000 | 0.733 / 0.861 / 0.673 | — |
| 30,000 | 0.716 / 0.851 / 0.657 | — |

Final, whole test sets, greedy CTC decoding (WER / CER):

| arm | params | updates | test-clean | test-other | Kokoro |
|---|---|---|---|---|---|
| plain | 3.30 M | 30,000 | 0.757 / 0.293 | 0.881 / 0.435 | 0.662 / 0.232 |
| MLP | 5.71 M | 10,000 | **0.562 / 0.190** | **0.759 / 0.332** | **0.492 / 0.158** |

Checkpoints on the lab, `~/zara/stt/runs2/runs/<arm>/params.pt`: plain md5
2bbc3d3f21f3678bf05c4e7fce2cb5fc, MLP 654109dba9681701c9c3ad3385fd7c9b.
Learned frame-rate decays stayed spread from ~15 ms to 2.6–5.3 s in both arms.

## Decided vs measured

Decided: the parameter and multiply-add counts. Measured, one seed: every WER; the
equal-step gap includes the schedule difference above.

## Verdict

The prediction held on both counts. The plain arm ended at 0.716 on the subset (inside
65–75%). The MLP moved step 5,000 from 0.817 to 0.626, well past the threshold, and with a
third of the updates it beat the full plain run by 19 points on test-clean. The missing
pointwise nonlinearity was the bottleneck. One seed, and the schedule confound inflates the
equal-step gap somewhat; it cannot explain a 19-point lead over a run with three times the
updates.

## What it changes

The recogniser keeps the MLP. Round 3 compares the fronts again on this backbone at 30,000
updates each: [stt-front-osc-vs-mel-mlp](../2026-10-23-stt-front-osc-vs-mel-mlp/). Decoding
with a word language model is measured separately:
[stt-ngram-decoding](../2026-10-23-stt-ngram-decoding/).
