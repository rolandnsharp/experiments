# Input-dependent rotation of the polar pole

**Record.** EXP id pending: experiments.blah.dev was unreachable when this protocol was written
(2026-10-24 ~10:00 UTC). This file, committed before the first run was launched, is the
pre-registration; the site record is created from it once the server is back, and its
methodology states the dates.

**Date.** 2026-10-24. **Status.** pre-registered (in git), runs not yet launched.

## Question

The polar pole c = exp(-(gammaMin + e^rho)) e^(i omega) rotates by a fixed angle omega per
byte. Does letting the input set the angle, theta_t = omega + W u_t, with the damping kept
fixed, lower held-out bpb?

## Prior art

- Mamba-3 ([Lahoti et al. 2026, 2603.15569](https://arxiv.org/abs/2603.15569), ICLR 2026):
  a complex-valued SSM implemented as data-dependent rotary embeddings; complex beats
  real-valued at the same state, halves the state for equal perplexity, and solves parity
  and modular arithmetic where real-valued models do not.
- Negative eigenvalues ([Grazzi et al. 2024, 2411.12537](https://arxiv.org/abs/2411.12537)):
  real decays in [0, 1] cannot track state; rotations can.
- LRU (fixed polar form, ours), selective-damping-4l (the sister screen: the damping
  input-dependent, the angle fixed).
- Borrowed: the input-dependent rotation. Ours: the exp-exp damping kept fixed so the
  modulus stays below 1, the rotation added to a learned constant frequency bank, at
  byte level, against Diakosmos's real decay.

## Hypothesis and predictions

Written before any run.

1. `--phase` scores **0.01–0.04 bpb below** the control, as the mean over seeds 0 and 1,
   and below it on each seed. A mean gain under 0.01, or worse on either seed, refutes it.
2. Step time within 1.5x the control's (the same chunked path as --select).
3. Recorded, not predicted: its gain against the selective-damping arm's.

## Variables

Independent: the pole angle, fixed (control) or input-dependent
(`temporal.attn.pole_omega_select`, (H*L, D), zero at the start; 3,486,040 parameters,
+3.0%, the same as --select). `LEGS="polar:dense:3000:0 polar:dense:3000:1" LAYERS=4
HARMONICS=32 EXTRA="--phase" TAG=phase`.

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
