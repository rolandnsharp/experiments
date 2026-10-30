# Speech recognition: Muon on our block's matrices vs AdamW

**Date.** 2026-10-29. **Status.** concluded, **supported**, [EXP-0051](https://experiments.blah.dev/EXP-0051),
pre-registered 15:03 UTC before launch (launched while its control, EXP-0050, was still training, on
Roland's decision). One A40, 9.0 pod-hours, $5.30.

## Question

With learning rate and warmup matched (3e-3, 200), does Muon on the blocks' hidden matrices close the
gap between our block and round 3's?

## Method

Muon written from the published descriptions (Jordan et al. 2024; Liu et al. 2025):
`harness/stt2-muon/zara_block.py` (`orthogonalize`, `Muon`, `Optimizer(muon=True)`). Nesterov momentum
0.95, quintic Newton–Schulz (3.4445, −4.7750, 2.0315, 5 steps), update scaled by 0.2 sqrt(max(rows, cols))
so it shares AdamW's rate, decoupled weight decay 0.01. Muon takes the 2-D tensors under `blocks.*`
(in, out, gate, proj, MLP w1/w3/w2); AdamW keeps the front projection, the head, the poles (half rate),
norms, biases and skips. Everything else as EXP-0050 (`launch-muon.sh`).

Checks before launch (not results): orthogonalised updates have singular values 0.68–1.13 on rectangular
matrices; a toy fit trains; optimizer state round-trips; a 600-update CPU run on dev-clean
(`cpu-muon.sh`) trains without diverging (mean loss 2.451 vs AdamW 2.471, both on the blank plateau).

## Prediction

Whole test-clean greedy WER 0.36–0.42, and at least 0.03 better than EXP-0050.

## Results

| whole sets | Muon | AdamW (EXP-0050) | round 3 |
|---|---|---|---|
| test-clean, greedy | **0.338** (CER 0.102) | 0.491 | 0.386 |
| test-other, greedy | 0.577 (CER 0.229) | 0.698 | 0.621 |
| Kokoro, greedy | 0.354 (CER 0.106) | 0.454 | 0.370 |
| test-clean, int8 C port, greedy | 0.340 (CER 0.103) | 0.502 | |
| test-clean, int8, beam 32 + 3-gram | 0.173 (CER 0.070) | 0.276 | |
| test-clean, int8, beam 64 + 3-gram | **0.165** (CER 0.066) | 0.259 | 0.194 |
| A53 RTF, int8, one core | **0.243** | | 0.256 |
| training s/update (A40) | 0.760 | 0.748 | |

alpha 0.5, beta 1.0 chosen on dev-clean 300 (dev WER 0.184). Timing: 12 test-clean files, chunk 16,
three alternating passes in one session (Muon 0.253 / 0.234 / 0.241; round 3 0.275 / 0.245 / 0.249).

Monitoring subset (200 test-clean), Muon / AdamW / round 3: 999 0.889 / 1.063 / 1.065;
1,999 0.753 / 0.971 / 0.885; 2,999 0.669 / 0.898 / 0.787; 7,999 0.502 / 0.716 / 0.562;
14,999 0.396 / 0.572 / 0.434; 22,999 0.315 / 0.491 / 0.369; 29,999 0.286 / 0.436.

## Verdict

**Supported.** Muon beat AdamW by 0.152 (predicted ≥ 0.03) and passed round 3 by 0.048 greedy and
0.029 with the LM, at equal size and budget, while staying ~5% faster on the A53. The absolute range
(0.36–0.42) is missed on the better side. The earlier lag (EXP-0048, EXP-0050) was the optimizer, not
the block. Muon becomes the optimizer for the release run (EXP-0049). One seed per arm; Muon's settings
taken from the literature, not tuned.
