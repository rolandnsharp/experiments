# Speech recognition, round 6: the 10.4 M model trained five times longer

**Date.** 2026-10-29. **Status.** running, [EXP-0043](https://experiments.blah.dev/EXP-0043),
pre-registered 2026-10-28 22:08 UTC before launch. Replaces the deferred size comparison
([stt-capacity](../2026-10-24-stt-capacity/), EXP-0023) for budget: Roland's balance ($56.68)
covered one of the two.

## Question

Round 3 (5.8 M, 30,000 updates, ~2.3 passes over 886 h) was still improving. Trained from scratch
in one schedule of 150,000 updates (~11.5 passes), how far does a 10.4 M version (6 blocks, MLP
672, width 448) get, and is it good enough to release?

## Prediction

| | round 3 | predicted |
|---|---|---|
| test-clean greedy | 0.386 | 0.22–0.28 |
| test-clean, beam 64 + 3-gram, dev-tuned | 0.194 | 0.10–0.14 |
| test-other greedy | 0.621 | 0.40–0.50 |

Worse than 0.30 greedy: length and this size step are not the main limit. Better than 0.20:
train longer still. Release criterion, fixed now: publish on Hugging Face if beam 64 + 3-gram
≤ 0.14 and A53 RTF < 0.40 (the shape measured 0.305 with random weights).

## Method

osc front, 6 blocks, MLP 672, 150,000 updates, batch 240 s, round-3 optimizer with warmup and
cosine over 150,000 updates, seed 0, no augmentation; subset WER every 2,500 updates (monitoring
only). Size and length move together: the gain is attributed to neither alone.
`launch-round6.sh`, `pod-stt6.sh`, `retry6.sh`. Pod `roland-stt7-a40-1` (A40, $0.59/h; no 4090 free).

## Results, Verdict

Pending.
