#!/usr/bin/env python3
"""Checks for --conv, --gate and --phase (CPU): the recurrence with a rotation per position against a
position-by-position loop and, with a constant rotation, against the constant-rotation path; each
flag the identity at the start; the chunked forward against the Sampler with all three on and their
weights random; optimizer roles; parameter counts."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diakosmos_oracle as O  # noqa: E402
import zara_poles as Z  # noqa: E402

torch.manual_seed(0)
B, T, H, L, dh = 2, 128, 3, 4, 5
qh = torch.randn(B, T, H, 2 * L, dtype=torch.float64)
kh = torch.randn(B, T, H, L, dtype=torch.float64)
v = torch.randn(B, T, H, dh, dtype=torch.float64)
la = -(0.01 + torch.rand(B, T, H, L, dtype=torch.float64))
th = torch.randn(B, T, H, L, dtype=torch.float64)
Sre = torch.zeros(B, H, L, dh, dtype=torch.float64); Sim = torch.zeros_like(Sre)
reads = []
for t in range(T):
    cre, cim = torch.exp(la[:, t]) * torch.cos(th[:, t]), torch.exp(la[:, t]) * torch.sin(th[:, t])
    inj = kh[:, t, :, :, None] * v[:, t, :, None, :]
    Sre, Sim = cre[..., None] * Sre - cim[..., None] * Sim + inj, cre[..., None] * Sim + cim[..., None] * Sre
    reads.append(torch.einsum('bhl,bhld->bhd', qh[:, t, :, :L], Sre) + torch.einsum('bhl,bhld->bhd', qh[:, t, :, L:], Sim))
loop = torch.stack(reads, 1)
d = (Z.recurrence_selective(qh, kh, v, la, th) - loop).abs().max().item()
print(f'(1) rotation per position vs the position loop: max |diff| {d:.2e} (scale {loop.abs().max().item():.1f})')
tc = th[0, 0]
d = (Z.recurrence_selective(qh, kh, v, la, tc.expand(B, T, H, L)) - Z.recurrence_selective(qh, kh, v, la, tc)).abs().max().item()
print(f'(2) a constant rotation given per position vs as [H][L]: max |diff| {d:.2e}')

c = O.Config(harmonics=4, layers=2, heads=7, kan=8, dt=0.3)
table = O.wire_table(c.harmonics)
w = torch.randint(0, 256, (2, 129))
p0 = Z.make_params(c, 'polar', 0)
l0 = Z.loss_of(c, p0, table, w, 'polar').item()
for name, kw, new in (('conv 4', dict(conv=4), ('qkv_conv',)), ('gate 16', dict(gate=16), ('out_gate',)),
                      ('phase', dict(phase=True), ('pole_omega_select',)),
                      ('select + phase', dict(select=True, phase=True), ('pole_decay_select', 'pole_omega_select'))):
    p1 = Z.make_params(c, 'polar', 0, **kw)
    base = {k: t for k, t in p1.items() if not any(n in k for n in new)}
    same = p0.keys() == base.keys() and all(torch.equal(p0[k], base[k]) for k in p0)
    l1 = Z.loss_of(c, p1, table, w, 'polar').item()
    print(f'(3) --{name} at init: other parameters identical {same}; loss diff {abs(l0 - l1):.1e}')
pm = Z.make_params(c, 'polar', 0, mlp=24, select=True, conv=4, gate=16, phase=True)
g = torch.Generator().manual_seed(5)
for k in pm:
    if any(n in k for n in ('select', 'qkv_conv', 'out_gate_b')):
        pm[k] = torch.randn(pm[k].shape, generator=g) * 0.3
with torch.no_grad():
    logits = Z.forward(c, pm, table, w[:1, :64], 'polar')[0]
    s = Z.Sampler(c, pm, table, 'polar')
    sl = torch.stack([s.logits(int(b)) for b in w[0, :64]])
print(f'(4) all flags, new weights random: chunked forward vs Sampler max |diff| {(logits - sl).abs().max().item():.2e}')
g0 = {k: t for k, t in pm.items() if 'select' in k or 'qkv_conv' in k or 'out_gate' in k}
pm = {k: t.requires_grad_() for k, t in pm.items()}
Z.loss_of(c, pm, table, w, 'polar').backward()
print('    gradients nonzero:', {k.split('.', 3)[-1]: bool(pm[k].grad.abs().sum() > 0) for k in g0 if k.startswith('blocks.0.')})
print('    roles:', {k.split('.', 3)[-1]: O.role(k, t.shape) for k, t in g0.items() if k.startswith('blocks.0.')})


def count(h, **kw):
    cc = O.Config(harmonics=h, layers=4, heads=7, kan=8, dt=0.3)
    return sum(t.numel() for t in Z.make_params(cc, 'polar', 0, **kw).values())
print('(5) parameters, 4 layers: polar h32', count(32), '| conv 4', count(32, conv=4), '| gate 16', count(32, gate=16),
      '| phase', count(32, phase=True), '| select', count(32, select=True))
