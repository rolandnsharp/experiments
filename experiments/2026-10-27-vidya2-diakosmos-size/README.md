# Vidya-2 at Diakosmos's size: against his 1.4194

**Record.** [EXP-0034](https://experiments.blah.dev/EXP-0034) on experiments.blah.dev (pre-registered
2026-10-26 14:19 UTC lab clock, before any run); the protocol and predictions are there, and the numbers
there and here match.

**Status.** stage 1 (updates 0–640) launched on a community RTX 4090, Roland's account (MARKS v2).

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

(pending)
