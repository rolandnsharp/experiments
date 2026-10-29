# Speech recognition: our own block and optimizer vs the round-3 block

**Date.** 2026-10-29. **Status.** concluded, **refuted** (accuracy), [EXP-0048](https://experiments.blah.dev/EXP-0048),
pre-registered 2026-10-29 01:14 UTC before launch. First pod crashed before update 0 (CUDA device
mismatch in a log line, $0.25, fixed); rerun 02:21–09:00 UTC on an A40, 6.7 pod-hours, $3.95.

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
And with |c| < 1 and inputs bounded by M, the streamed state h(n+1) = c h(n) + g u(n) from h(0) = 0
never exceeds |g| M / (1 - |c|), however long the stream runs, so the C port cannot overflow:
[zara-block-state-bounded](https://proofs.blah.dev/theorems/01a18580-38c9-77ae-b967-a11aacdb9f88)
(induction: the bound is the fixed point of x -> |c| x + |g| M).

## Prediction

Whole test-clean greedy WER **0.34–0.42** (round 3: 0.386). int8 C port A53 RTF ≤ 0.30
(round 3: 0.245). Worse than 0.42: look at the untuned learning rate (1.5e-3) before the block.

## Method

Width 448, 4 blocks, 224 complex states each, MLP 448 (5.73 M parameters vs 5.78 M); osc front;
30,000 updates of 240 s; AdamW (0.9/0.98, wd 0.01 on matrices, poles at half rate), warmup
1,000, cosine to 10%, clip 1.0; seed 0; no augmentation. `harness/stt2/launch-check.sh`.
Scoring as rounds 3–4 (`port/tune-model3.sh` on the lab: format-3 log-probs, the same LM decoder).

## Results

| whole sets | own block | round 3 |
|---|---|---|
| test-clean, greedy | **0.509** (CER 0.172) | 0.386 |
| test-other, greedy | 0.711 (CER 0.309) | 0.621 |
| Kokoro, greedy | 0.468 (CER 0.149) | 0.370 |
| test-clean, int8 C port, greedy | 0.509 (CER 0.173) | |
| test-clean, int8, beam 32 + 3-gram | 0.292 (CER 0.130) | |
| test-clean, int8, beam 64 + 3-gram | **0.282** (CER 0.123) | 0.194 |
| A53 RTF, int8, one core | **0.228** | 0.241 |

alpha 0.5, beta 2.0 chosen on dev-clean 300 (dev WER 0.307). Timing: 12 test-clean files, chunk 16,
three alternating passes in one session (own 0.211 / 0.242 / 0.231; round 3 0.225 / 0.255 / 0.243).

Monitoring subset (200 test-clean), own vs round 3: 999 1.095 vs 1.065; 1,999 1.064 vs 0.885;
3,999 0.918 vs 0.717; 7,999 0.741 vs 0.562; 11,999 0.654 vs 0.478; 15,999 0.571 vs 0.427;
22,999 0.491 vs 0.369.

## Verdict

**Refuted** on accuracy (0.509 against 0.34–0.42); speed met (0.228 ≤ 0.30). The curve kept round
3's shape but stretched (its WER at 8,000 updates was round 3's at ~4,000), which points at
optimisation speed: AdamW at 1.5e-3 with a 1,000-update warmup against round 3's 3e-3, 200-update
warmup and a Muon-type update. Block and optimizer changed together (declared confound), so this
does not show the block is worse. Next, per the analysis plan: the learning rate,
[EXP-0050](https://experiments.blah.dev/EXP-0050) (3e-3, warmup 200; drafted before these final
scores). int8 costs nothing; the block is 5% faster than round 3's on the A53.
