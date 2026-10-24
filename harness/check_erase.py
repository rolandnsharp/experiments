#!/usr/bin/env python3
"""Checks for --erase (CPU): recurrence_erase against a position-by-position loop of
S_t = (I - beta k^ k^') c_t S_(t-1) + k v', with damping and rotation per position; with beta = 0 against
recurrence_selective; the start against the control; the chunked forward against the Sampler with
erase, select and phase on and random weights; gradients and optimizer roles; parameter count."""
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
beta = torch.rand(B, T, H, dtype=torch.float64)
S = torch.zeros(B, H, L, dh, dtype=torch.complex128)
reads = []
kn = kh / (kh.norm(dim=-1, keepdim=True) + 1e-6)
q = torch.complex(qh[..., :L], qh[..., L:])
for t in range(T):
    X = torch.exp(torch.complex(la[:, t], th[:, t]))[..., None] * S
    X = X - beta[:, t, :, None, None] * kn[:, t, :, :, None] * torch.einsum('bhl,bhld->bhd', kn[:, t].to(X.dtype), X)[:, :, None, :]
    S = X + (kh[:, t, :, :, None] * v[:, t, :, None, :]).to(X.dtype)
    reads.append(torch.einsum('bhl,bhld->bhd', q[:, t].conj(), S).real)
loop = torch.stack(reads, 1)
d = (Z.recurrence_erase(qh, kh, v, la, th, beta) - loop).abs().max().item()
print(f'(1) erase vs the position loop (beta random in (0,1)): max |diff| {d:.2e} (scale {loop.abs().max().item():.1f})')
d = (Z.recurrence_erase(qh, kh, v, la, th, torch.zeros_like(beta)) - Z.recurrence_selective(qh, kh, v, la, th)).abs().max().item()
print(f'(2) beta = 0 vs recurrence_selective: max |diff| {d:.2e}')
print(f'    state norm, last position, beta random vs beta 0: {S.abs().max().item():.2f}')

c = O.Config(harmonics=4, layers=2, heads=7, kan=8, dt=0.3)
table = O.wire_table(c.harmonics)
w = torch.randint(0, 256, (2, 129))
p0 = Z.make_params(c, 'polar', 0)
p1 = Z.make_params(c, 'polar', 0, erase=True)
base = {k: t for k, t in p1.items() if 'erase_gate' not in k}
same = p0.keys() == base.keys() and all(torch.equal(p0[k], base[k]) for k in p0)
l0 = Z.loss_of(c, p0, table, w, 'polar').item(); l1 = Z.loss_of(c, p1, table, w, 'polar').item()
print(f'(3) --erase at the start: other parameters identical {same}; loss {l0:.6f} vs {l1:.6f} (diff {abs(l0 - l1):.1e})')
pm = Z.make_params(c, 'polar', 0, select=True, phase=True, erase=True)
g = torch.Generator().manual_seed(5)
for k in pm:
    if 'select' in k or 'erase_gate_weight' in k:
        pm[k] = torch.randn(pm[k].shape, generator=g) * 0.3
    if 'erase_gate_bias' in k:
        pm[k] = torch.zeros_like(pm[k])
with torch.no_grad():
    logits = Z.forward(c, pm, table, w[:1, :64], 'polar')[0]
    s = Z.Sampler(c, pm, table, 'polar')
    sl = torch.stack([s.logits(int(b)) for b in w[0, :64]])
print(f'(4) erase + select + phase, random: chunked forward vs Sampler max |diff| {(logits - sl).abs().max().item():.2e}')
pm = {k: t.requires_grad_() for k, t in Z.make_params(c, 'polar', 0, erase=True).items()}
Z.loss_of(c, pm, table, w, 'polar').backward()
print('    gradients nonzero:', {k.split('.', 3)[-1]: bool(pm[k].grad.abs().sum() > 0) for k in pm if k.startswith('blocks.0.') and 'erase' in k})
print('    roles:', {k.split('.', 3)[-1]: O.role(k, t.shape) for k, t in pm.items() if k.startswith('blocks.0.') and 'erase' in k})
cc = O.Config(harmonics=32, layers=4, heads=7, kan=8, dt=0.3)
print('(5) parameters, 4L/32H: polar', sum(t.numel() for t in Z.make_params(cc, 'polar', 0).values()),
      '| erase', sum(t.numel() for t in Z.make_params(cc, 'polar', 0, erase=True).values()))
