# Speech recognition: our own block and optimizer vs the round-3 block

**Date.** 2026-10-29. **Status.** running, [EXP-0048](https://experiments.blah.dev/EXP-0048),
pre-registered 2026-10-29 01:14 UTC before launch. Pod `roland-stt8-a40-1` (A40, $0.59/h).

## Question

The release must contain only this project's code. The frame-rate block and optimizer were
rewritten from the open literature (`harness/stt2/zara_block.py`): per block, N continuous-time
complex poles between two linear maps,

    u = LayerNorm(z);  h_t = c h_{t-1} + gamma (B u_t);  y = C [Re h; Im h] + d * u;
    z <- z + W_o (y * silu(W_g u)) + b_o,   c = exp(-dt/tau + i 2 pi f dt),  gamma = sqrt(1 - |c|^2)

(diagonal complex recurrences: S4D, LRU; gated output: GLU/Mamba), trained with AdamW, warmup and
cosine. At round 3's shape and budget, is it as accurate as the block it replaces?

## Checks before the run (not results)

- chunked scan = plain loop (max diff 1.1e-12); block in parallel = frame by frame (4.4e-16)
- C port (model format 3) = Python on the same random weights: max log-prob diff 1.3e-5,
  argmax agrees on every frame, chunked (K = 16) and streaming (K = 1)
- a 600-update CPU run on dev-clean (`cpu-compare.sh`) trains without diverging (one loss
  spike at the end of its 60-update warmup, recovered by update 80)

## Proved, not measured

Every pole has |c| < 1 for every value of log_tau and theta, so no training step can make the
recurrence unstable: [zara-block-pole-inside-unit-circle](https://proofs.blah.dev/theorems/01a1853d-f715-74a9-ba73-a2b2bb5b7ac0),
verified in Lean on proofs.blah.dev (|exp z| = exp(Re z) and Re z = -(1/TAU_MAX + e^(-x))/rate < 0).

## Prediction

Whole test-clean greedy WER **0.34–0.42** (round 3: 0.386). int8 C port A53 RTF ≤ 0.30
(round 3: 0.245). Worse than 0.42: look at the untuned learning rate (1.5e-3) before the block.

## Method

Width 448, 4 blocks, 224 complex states each, MLP 448 (5.73 M parameters vs 5.78 M); osc front;
30,000 updates of 240 s; AdamW (0.9/0.98, wd 0.01 on matrices, poles at half rate), warmup
1,000, cosine to 10%, clip 1.0; seed 0; no augmentation. `harness/stt2/launch-check.sh`.
Scoring as rounds 3–4 (`tune-model.sh` with the format-3 exporter).

## Results, Verdict

Pending.
