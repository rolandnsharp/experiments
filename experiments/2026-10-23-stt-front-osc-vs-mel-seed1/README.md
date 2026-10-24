# Speech recognition: oscillator front vs mel, second seed

**Date.** 2026-10-23. **Status.** first attempt failed before training (disk full); not yet rerun. Prediction written before the run.

## Question

[stt-front-osc-vs-mel-mlp](../2026-10-23-stt-front-osc-vs-mel-mlp/) found the learned oscillator
front ahead of mel on all three test sets with one seed (test-clean 0.386 vs 0.402). Does the
ordering hold with a different seed?

## Prior art

As in the seed-0 paper (SincNet, LEAF: learnable front ends roughly match mel). The seed-0 run
itself is the only evidence for "better".

## Hypothesis and prediction

Written 2026-10-23 before the run: the oscillator front beats mel again on test-clean, by 0.5–3
points greedy (seed 0: 1.6); both arms land within ±2 points of their seed-0 numbers (osc
0.37–0.41, mel 0.38–0.42). What would change our mind: mel ahead, or the gap under 0.5 points,
means the seed-0 gap was noise and the card says "matches mel", not "beats".

## Independent variable

The seed (1 instead of 0): initialisation, batch order and every other draw. Everything else as
seed 0: osc (64 poles) and mel (40 bands) fronts, 4 blocks, width 448, gated MLP 448, 30,000
updates, batches of 240 s, 885.8 h of LibriSpeech, no augmentation. The hardware differs (RTX 3090
instead of A40); results should not depend on it beyond floating-point order.

## Code

`harness/stt/zara_stt.py` (`--seed 1`), `pod-stt5.sh`, `retry5.sh` (3090 only), `pod-run-all.sh`.

## Hardware and cost

RunPod community RTX 3090, $0.22/h, estimated 12–14 h ≈ $3.

## Results, Decided vs measured, Verdict, What it changes

Pending.

### Attempt 1 (failed before training, 2026-10-23 11:50 to 2026-10-24 07:25 UTC)

Pod `roland-stt5-3090-3` (RTX 3090, $0.22/h) was given a 70 GB disk to save money. Round 4 had
ended at 53 GB used on 100 GB, but the peak is higher: while train-other-500 is being packed, its
extracted FLAC files (~30 GB) and the growing packed file (28.6 GB) sit beside the other packed
subsets (27 GB). The disk filled at 16:24 with train-other-500 half packed (15.4 of 28.6 GB); the
128 packer processes then blocked on writes and never exited, so the driver kept waiting. Found and
removed 2026-10-24 07:25. No training ran; nothing to score. Cost ≈ $4.30 for nothing.

Fixes for the rerun: disk 100 GB, and the driver should treat "no change in the data log for an
hour" as a failure and stop the pod.
