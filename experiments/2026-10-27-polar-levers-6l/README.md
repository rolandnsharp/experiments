# Polar-model training levers at 6 layers

**Record.** [EXP-0035](https://experiments.blah.dev/EXP-0035) on experiments.blah.dev (pre-registered
2026-10-27 04:11 UTC lab clock, before any run); the numbers there and here match.

**Status.** concluded 2026-10-27, supported. 8 legs on community 3090/3080 pods (MARKS po-ctl, po-lr2,
po-wide, po-polelr4), about $4.9 in all.

## Why

The polar pole is the only change measured to help at Diakosmos's size (EXP-0011: 0.104 at update 640);
the block changes screened at 4 layers did not carry (EXP-0034). So the next candidates act on the
polar model's training, not its block, and are screened at 6 layers first. The poles start with
Diakosmos's decays, 7.9–20.6 bytes, and no long memory; trained, most decay in 1–4 bytes and a few in
hundreds (4L, 30K updates).

## Arms (polar 6L/64H, 3,000 updates, Diakosmos's recipe, seeds 0 and 1)

| arm | change |
|---|---|
| control | none (`--pole-lr-scale=1`, the default path, so the leg string carries a name) |
| lr2 | peak learning rate x2 |
| wide | decays log-spaced 1–1500 bytes at the start (`--pole-init wide`) |
| polelr4 | poles move 4x faster (`--pole-lr-scale 4`: stored over 4) |

Predicted before any run: lr2 0.01–0.04 below the control; wide 0.01–0.03; polelr4 0.005–0.02. A pass
(mean ≥ 0.01, below on both seeds) earns a 640-update gate at 100 M.

## Results

All 18,538,212 held-out bytes:

| arm | seed 0 | seed 1 | mean | gain | prediction |
|---|---|---|---|---|---|
| control | 1.8804 | 1.8783 | 1.8794 | - | - |
| lr2 | 1.8520 | 1.8542 | 1.8531 | -0.0263 | met (pass) |
| wide | 1.8688 | 1.8703 | 1.8696 | -0.0098 | missed by 0.0002 |
| polelr4 | 1.8700 | 1.8668 | 1.8684 | -0.0110 | met (pass) |

Every arm is below the control on both seeds; the control's seeds differ by 0.002. Step time on fast
hosts: 1.87 s (control) and 1.93 s (lr2) an update. wide and polelr4 landed on slow hosts (4.6–5.3 s),
and their seed 1 legs were resumed from the lab's checkpoint on replacement pods (wide at update 1,000,
polelr4 at 500). No lever adds multiply-adds (20.34 M a byte).

## What it means

The polar model's inherited training settings leave gain on the table. The peak rate is the largest
lever, as at 4 layers (EXP-0010). The two pole levers give about 0.01 each; both help the poles leave
Diakosmos's 8–20 byte decays sooner, so they may be substitutes and are not assumed to add. lr2 earns its
640-update gate at 100 M (EXP-0036); polelr4 passes too and earns one. wide does not pass the bar.

Limits: two seeds, one size, 3,000 updates, one setting per lever, levers not combined.
