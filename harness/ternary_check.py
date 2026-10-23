#!/usr/bin/env python3
"""CPU check of --ternary (experiment 2026-10-23-ternary-projections), before any pod:

    ternary_check.py OLD_DIR

1. Flag off: the loss and every gradient on a fixed batch equal those of the
   harness in OLD_DIR (the files before the change), bit for bit.
2. Flag on: every qkv/out projection the forward uses takes three values times
   its scale, the gradient reaches the latent weights (straight through), and the
   loss differs from the float loss.
3. Flag on: the Sampler's position-by-position logits equal the chunked forward's.
"""
import importlib.util
import sys
from pathlib import Path

import torch

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import diakosmos_oracle as O  # noqa: E402
import zara_poles as Z  # noqa: E402


def load_old(d):
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location('zara_poles_old', Path(d) / 'zara_poles.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.path.pop(0)
    return m


def loss_and_grads(m, c, p0, table, w):
    p = {k: v.clone().requires_grad_(True) for k, v in p0.items()}
    l = m.loss_of(c, p, table, w, 'polar')
    l.backward()
    return l.detach(), {k: v.grad for k, v in p.items()}


def main():
    old = load_old(sys.argv[1])
    torch.manual_seed(0)
    c = O.Config(harmonics=8, layers=2, heads=7, kan=8, dt=0.3)
    table = O.wire_table(c.harmonics)
    p0 = Z.make_params(c, 'polar', 0)
    w = torch.randint(0, 256, (2, 129))

    l_old, g_old = loss_and_grads(old, c, p0, table, w)
    l_new, g_new = loss_and_grads(Z, c, p0, table, w)
    assert torch.equal(l_old, l_new), (l_old, l_new)
    assert all(torch.equal(g_old[k], g_new[k]) for k in g_old)
    print(f'1. flag off: loss {l_new.item():.6f} and {len(g_new)} gradients equal the old harness bit for bit')

    Z.TERNARY = True
    names = [k for k in p0 if k.endswith(('attn.qkv_proj.weight', 'attn.out_proj.weight'))]
    for k in names:
        q = Z.projection(p0, k)
        s = p0[k].abs().mean()
        assert set((q / s).round().unique().tolist()) <= {-1.0, 0.0, 1.0}
        assert torch.allclose(q, (q / s).round() * s)
    l_t, g_t = loss_and_grads(Z, c, p0, table, w)
    assert all(g_t[k].abs().sum() > 0 for k in names)
    assert not torch.equal(l_t, l_new)
    zeros = sum((Z.projection(p0, k) == 0).sum().item() for k in names) / sum(p0[k].numel() for k in names)
    print(f'2. flag on: {len(names)} projections ternary ({zeros:.1%} zeros at init); loss {l_t.item():.6f} vs float {l_new.item():.6f}; gradients reach the latent weights')

    with torch.no_grad():
        chunked = Z.forward(c, p0, table, w[:1, :64], 'polar')[0]
        s = Z.Sampler(c, p0, table, 'polar')
        stepped = torch.stack([s.logits(int(b)) for b in w[0, :64]])
    err = (chunked - stepped).abs().max().item()
    assert err < 1e-3, err
    print(f'3. flag on: Sampler equals the chunked forward (max |diff| {err:.2e} over 64 positions)')


if __name__ == '__main__':
    main()
