#!/usr/bin/env python3
"""CPU check of --ternary and --act8 (experiments 2026-10-23-ternary-projections and
-act8-projection-inputs), before any pod:

    ternary_check.py OLD_DIR

1. Flag off: the loss and every gradient on a fixed batch equal those of the
   harness in OLD_DIR (the files before the change), bit for bit.
2. Flag on: every qkv/out projection the forward uses takes three values times
   its scale, the gradient reaches the latent weights (straight through), and the
   loss differs from the float loss.
3. Flag on: the Sampler's position-by-position logits equal the chunked forward's.
4. --ternary on, --act8 off: the loss and every gradient equal OLD_DIR's ternary
   forward bit for bit (when OLD_DIR has one).
5. --act8 on: every projection input lies on its position's 255-level grid
   (max |x| / 127 times an integer in [-127, 127]), the loss moves, the gradient
   still reaches every parameter the float forward reaches, and the Sampler equals
   the chunked forward in float64 (in float32 their ~1e-6 differences flip an
   occasional rounding by a whole grid step).
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

    if hasattr(old, 'TERNARY'):
        old.TERNARY = True
        l_ot, g_ot = loss_and_grads(old, c, p0, table, w)
        assert torch.equal(l_ot, l_t) and all(torch.equal(g_ot[k], g_t[k]) for k in g_t)
        print(f'4. --ternary on, --act8 off: loss {l_t.item():.6f} and gradients equal the old ternary forward bit for bit')
    else:
        print('4. skipped: OLD_DIR predates --ternary')

    Z.ACT8 = True
    x = torch.randn(3, 5, c.d) * 3
    q = Z.act8(x)
    s = x.abs().amax(-1, keepdim=True)
    k = q * 127 / s
    assert torch.allclose(k, k.round(), atol=1e-3) and k.round().abs().max() <= 127
    l_a, g_a = loss_and_grads(Z, c, p0, table, w)
    assert not torch.equal(l_a, l_t)
    assert all(g_a[n].abs().sum() > 0 for n in g_t if g_t[n].abs().sum() > 0)
    # in float64: in float32 the two paths' ~1e-6 differences flip an occasional int8
    # rounding, a whole grid step, so equality of the functions is checked where they cannot
    p64, t64 = {n: v.double() for n, v in p0.items()}, table.double()
    with torch.no_grad():
        chunked = Z.forward(c, p64, t64, w[:1, :64], 'polar')[0]
        sm = Z.Sampler(c, p64, t64, 'polar')
        stepped = torch.stack([sm.logits(int(b)) for b in w[0, :64]])
    err = (chunked - stepped).abs().max().item()
    assert err < 1e-9, err
    print(f'5. --act8 on (with --ternary): inputs on the 255-level grid, loss {l_a.item():.6f} vs {l_t.item():.6f}, gradients reach every parameter, Sampler equals chunked in float64 (max |diff| {err:.2e})')


if __name__ == '__main__':
    main()
