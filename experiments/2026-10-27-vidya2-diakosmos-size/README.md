# Vidya-2 at Diakosmos's size: against his 1.4194

**Record.** [EXP-0034](https://experiments.blah.dev/EXP-0034) on experiments.blah.dev (pre-registered
2026-10-26 14:19 UTC lab clock, before any run); the protocol and predictions are there, and the numbers
there and here match.

**Status.** concluded 2026-10-27: **refuted** at stage 1; the full run was not made.

## The block

Polar pole (Vidya.Pole) + a SwiGLU MLP after each block + a residual causal convolution of width 4 on
q, k, v + a rank-16 output gate on the read, at 14 layers, 60 harmonics (width 840) and MLP 1697:
100,505,412 parameters (polar 14L/95H 99,974,252; Diakosmos 99,823,934). Chosen from the 4-layer screens
(EXP-0024 to EXP-0028), the MLP split (EXP-0030: MLP about 60% of the parameters) and the factorial
(EXP-0029); input-dependent damping (2.2x the step time for ~0.003) and rotation (harmful in
combination) left out (Roland, 2026-10-27). Recipe exactly as EXP-0011's polar arm.

## Predictions (before any run)

1. At update 640: 1.70–1.83 on the full held-out set (polar 1.8794; Diakosmos 1.9806). 1.86 or above
   refutes it and stage 2 is not run.
2. At 15,360: 1.33–1.39, beating Diakosmos's 1.4194 and GPT-2 small's 1.4417. Above 1.41 refutes it.
3. Step time on a 4090 below the polar arm's 8.1 s.

## Results

Stage 1 ran on a secure RTX 4090 (deviation: no community 4090 was free), 14:34–16:41 UTC, $1.92.
Held-out bpb over all 18,538,212 scored bytes at update 640 of 15,360:

| model | parameters | bpb at 640 | step time (4090) |
|---|---|---|---|
| Diakosmos, real decay (EXP-0011 reproduction; his run 1.9806) | 99,823,934 | 1.9829 | — |
| polar pole (EXP-0011) | 99,974,252 | 1.8794 | 8.1 s |
| **Vidya-2** | 100,505,412 | **1.8869** | 10.8 s |

Training loss, polar / Vidya-2: 100: 3.381 / 3.399; 200: 2.343 / 2.348; 300: 2.068 / 2.083; 400: 1.999 /
2.015; 500: 1.988 / 2.000; 620: 1.927 / 1.928.

## Verdict

Prediction 1 missed (1.8869, the refutation line was 1.86); prediction 2 untested (stage 2 not run, per
the rule); prediction 3 missed (10.8 s against 8.1 s). The 4-layer gains (about 0.13 at 3.4 M
parameters) do not carry to 100 M with Diakosmos's recipe: the most likely reading is that the small
polar block was starved of per-byte nonlinear computation that an MLP and gate supplied, while at 14
layers and width 1330 the polar block already has it. Not excluded with one seed and 640 updates: a
later gain, a learning rate or MLP share that is wrong at this size. The harness is host-bound, so the
extra modules cost time although the model is narrower. Method lesson: an intermediate-size check
before 100 M; this stage-1 gate cost $1.92 instead of a ~$16 full run.
