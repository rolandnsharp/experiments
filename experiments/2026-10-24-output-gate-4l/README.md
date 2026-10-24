# An output gate on the recurrence's read

**Record.** EXP id pending: experiments.blah.dev was unreachable when this protocol was written
(2026-10-24 ~10:00 UTC). This file, committed before the first run was launched, is the
pre-registration; the site record is created from it once the server is back, and its
methodology states the dates.

**Date.** 2026-10-24. **Status.** pre-registered (in git), runs not yet launched.

## Question

Does scaling the recurrence's read channel by channel by an input-dependent gate,
read * 2 sigmoid(u A' B') with A (16, D), B (D, 16) and u the block's normalized input,
before the out projection, lower held-out bpb of the polar model?

## Prior art

- GLA ([Yang et al. 2023](https://arxiv.org/abs/2312.06635)), Gated DeltaNet, Kimi Linear
  (ablation: the sigmoid output gate critical) and Mamba (SiLU output gate) gate the read.
  Gated attention ([Qiu et al. 2025, 2505.06708](https://arxiv.org/abs/2505.06708)): a
  sigmoid gate after attention improves 1.7B and 15B models, stabilizes training and
  tolerates larger learning rates.
- Diakosmos has a scalar sigmoid gate between the temporal and spatial paths per group,
  not on the read.
- Borrowed: the elementwise sigmoid output gate; low rank 16 as GLA's gates. Ours: the
  2 sigmoid form, the identity at the start, and the measurement.

## Hypothesis and predictions

Written before any run.

1. `--gate 16` scores **0.01–0.04 bpb below** the control, as the mean over seeds 0 and 1,
   and below it on each seed. A mean gain under 0.01, or worse on either seed, refutes it.
2. Step time within 1.2x the control's.

## Variables

Independent: the output gate, absent (control) or rank 16 (`temporal.attn.out_gate_a`
random, `out_gate_b` zero; 3,443,032 parameters, +1.7%).
`LEGS="polar:dense:3000:0 polar:dense:3000:1" LAYERS=4 HARMONICS=32 EXTRA="--gate 16" TAG=gate16`.

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

(pending)
