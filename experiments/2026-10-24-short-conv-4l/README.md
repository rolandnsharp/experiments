# A short causal convolution on q, k, v

**Record.** [EXP-0026](https://experiments.blah.dev/EXP-0026) on experiments.blah.dev; the numbers there and here match. This protocol was committed to git before launch while the site was unreachable; the site record was created from it and pre-registered at 10:02–10:04 UTC, and its methodology states which runs had started by then.

**Date.** 2026-10-24. **Status.** concluded 2026-10-24: supported.

## Question

Does adding a causal depthwise convolution of width 4 to the query, key and value
projections (each channel its own 4 taps, added to the projection: qkv + conv(qkv)) lower
held-out bpb of the polar model?

## Prior art

- Mamba, Gated DeltaNet ([Yang et al. 2024](https://arxiv.org/abs/2412.06464): the
  short convolution and output gate "crucial" in ablation), Kimi Linear
  ([2510.26692](https://arxiv.org/abs/2510.26692)) and RWKV (token shift) all put a short
  convolution before the recurrence. Canon layers ([Allen-Zhu 2025,
  2512.17351](https://arxiv.org/abs/2512.17351)): residual short convolutions bring linear
  attention close to Mamba-2 and Gated DeltaNet. Based ([Arora et al. 2024,
  2402.18668](https://arxiv.org/abs/2402.18668)) found only small language-modelling gains.
- Diakosmos has no convolution. Our polar model's fastest learned poles have periods of
  about 2.3 bytes (text-pole-form), which may already supply the local mixing.
- Borrowed: the residual short convolution (Canon's form). Ours: the measurement here.

## Hypothesis and predictions

Written before any run.

1. `--conv 4` scores **0.01–0.05 bpb below** the control, as the mean over seeds 0 and 1,
   and below it on each seed. A mean gain under 0.01, or worse on either seed, refutes it.
2. Step time within 1.2x the control's.

## Variables

Independent: the convolution, absent (control, 3,385,688 parameters) or width 4
(`blocks.i.temporal.attn.qkv_conv_kernel`, (3D, 4), 3,407,192 parameters, +0.6%).
`LEGS="polar:dense:3000:0 polar:dense:3000:1" LAYERS=4 HARMONICS=32 EXTRA="--conv 4" TAG=conv4`.

## Protocol shared with the other screens

As [gated-mlp-4l](../2026-10-24-gated-mlp-4l/README.md): polar 4L/32H, Diakosmos's optimizer
and schedule for 3,000 updates at lr 1x, batch 16, 2 shards of OpenWebText bytes, seeds 0
and 1, scored on all 18,538,212 held-out bytes (the GPT-2 baseline's windows, never trained
on). Control (text-pole-form, same harness, bit-identical with the flags off): 2.0490
(seed 0), 2.0409 (seed 1). The new weights start so that the arm is the control (zero
kernel, zero gate factor or zero selection) and are trained with Adam. One community RTX
3090/3080 on Roland's account runs both seeds. Two seeds at 3K updates is a screen: a pass
earns longer and larger runs, not a claim.

## Code checks (lab CPU, `harness/check_conv_gate_phase.py`)

With the flags off the harness is bit for bit the previous one (`dev/test_harness.sh`); a
rotation per position equals a position-by-position loop to 2.6e-14 and, held constant,
the constant-rotation path to 4.7e-14; at the start `--conv` and `--gate` give the
control's loss exactly and `--phase` to 4.8e-7 (float32 rounding of a cumulative sum);
the chunked forward equals the one-position sampler with all flags on and random new
weights to 1.9e-5; every new weight gets a gradient and is in the Adam group.

## Results

| seed | control | conv 4 | gain |
|---|---|---|---|
| 0 | 2.0490 | 2.0072 | 0.042 |
| 1 | 2.0409 | 2.0013 | 0.040 |

Mean gain **0.041** (spread 0.002), 0.80 s an update against 0.75 s (1.07x), +0.6% parameters,
$0.15 + $0.15.

## Verdict

Both predictions met: the gain is inside 0.01–0.05 on both seeds, the step time within 1.2x. Third
of today's changes by size (MLP 0.114, gate 0.072); whether it adds beside the MLP is EXP-0029's
question (all five together 1.9333 against the MLP alone 1.9309).
