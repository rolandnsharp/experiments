# Speech recognition: oscillator front vs mel, second seed

**Date.** 2026-10-23. **Status.** running (one RTX 3090). Prediction written before the run.

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
