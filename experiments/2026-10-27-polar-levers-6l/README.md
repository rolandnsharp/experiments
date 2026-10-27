# Polar-model training levers at 6 layers

**Record.** [EXP-0035](https://experiments.blah.dev/EXP-0035) on experiments.blah.dev (pre-registered
2026-10-27 04:11 UTC lab clock, before any run); the numbers there and here match.

**Status.** running: 8 legs on 4 community 3090/3080 pods (MARKS po-ctl, po-lr2, po-wide, po-polelr4).

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

(pending)
