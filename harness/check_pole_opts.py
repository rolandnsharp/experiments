#!/usr/bin/env python3
"""Checks for --pole-lr-scale and --pole-init (CPU): with scale K the stored poles are the unscaled ones over K and the
forward is the same up to rounding; one optimizer update moves the effective rho and omega about K times as far as
without the scale; --pole-init wide gives decay lengths log-spaced 1-1500 bytes; flags off are checked separately by
dev/test_harness.sh (bit for bit)."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diakosmos_oracle as O  # noqa: E402
import zara_poles as Z  # noqa: E402

c = O.Config(harmonics=8, layers=2, heads=7, kan=8, dt=0.3)
table = O.wire_table(c.harmonics)
w = torch.randint(0, 256, (2, 129), generator=torch.Generator().manual_seed(0))
a = 'blocks.0.temporal.attn.'


def run(scale, init='diakosmos'):
    Z.POLE_SCALE = scale
    p = Z.make_params(c, 'polar', 0, pole_init=init)
    loss = Z.loss_of(c, p, table, w, 'polar')
    rho0, om0 = (t.clone() for t in Z.polar_params(p, a))
    opt = O.Update(p, total=1000)
    opt.step = 300                      # past warmup, so the learning rate is not zero
    p = {k: v.clone().requires_grad_() for k, v in p.items()}
    Z.loss_of(c, p, table, w, 'polar').backward()
    g = {k: v.grad for k, v in p.items()}
    q = {k: v.detach().clone() for k, v in p.items()}
    opt(q, g)                           # updates q in place
    rho1, om1 = Z.polar_params(q, a)
    Z.POLE_SCALE = 1.0
    return loss.item(), (rho1 - rho0).abs().median().item(), (om1 - om0).abs().median().item(), rho0


l1, dr1, do1, rho = run(1.0)
l4, dr4, do4, rho4 = run(4.0)
print(f'(1) scale 4 vs 1 at the start: loss {l4:.7f} vs {l1:.7f} (diff {abs(l4 - l1):.1e}); effective rho equal to {(rho4 - rho).abs().max().item():.1e}')
print(f'(2) one update: median |d rho| {dr4:.2e} vs {dr1:.2e} (ratio {dr4 / dr1:.2f}); median |d omega| {do4:.2e} vs {do1:.2e} (ratio {do4 / do1:.2f})')
lw, _, _, rhow = run(1.0, 'wide')
tau = 1 / (torch.exp(rhow) + Z.GMIN)
q = [round(float(v), 1) for v in torch.quantile(tau.flatten().double(), torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64))]
print(f'(3) --pole-init wide: decay lengths (min, quartiles, max) {q} bytes; loss at the start {lw:.4f} (default {l1:.4f})')
