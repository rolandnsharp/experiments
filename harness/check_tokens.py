#!/usr/bin/env python3
"""CPU check of --embed and --tokens (experiment 2026-10-27-bytes-vs-subword, EXP-0033), before any pod:

    check_tokens.py OLD_DIR

1. Flags off: on the factorial's best block (mlp, select, conv, gate), the loss and every gradient on a
   fixed batch equal those of the harness in OLD_DIR bit for bit.
2. --embed on bytes: the parameters are OLD_DIR's plus embed.harmonic_phasors (the head unchanged); each
   input row has unit length a group; the loss moves; the gradient reaches the table.
3. A 2048-token vocabulary: the head has 2048 rows, centred; the Sampler's position-by-position logits
   equal the chunked forward's (float64).
4. score_tokens: on toy windows, the bits per byte equal a hand sum of each window's token losses after
   the start token (by the Sampler), over the windows' bytes.
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import diakosmos_oracle as O  # noqa: E402
import field_poles  # noqa: E402
import zara_poles as Z  # noqa: E402

BEST = dict(mlp=24, select=True, conv=4, gate=4)


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
    c = O.Config(harmonics=4, layers=2, heads=7, kan=8, dt=0.3)
    table = O.wire_table(c.harmonics)
    w = torch.randint(0, 256, (2, 129), generator=torch.Generator().manual_seed(0))

    p_old = old.make_params(c, 'polar', 0, **BEST)
    p_new = Z.make_params(c, 'polar', 0, **BEST)
    assert p_old.keys() == p_new.keys() and all(torch.equal(p_old[k], p_new[k]) for k in p_old)
    l_old, g_old = loss_and_grads(old, c, p_old, table, w)
    l_new, g_new = loss_and_grads(Z, c, p_new, table, w)
    assert torch.equal(l_old, l_new), (l_old, l_new)
    assert all(torch.equal(g_old[k], g_new[k]) for k in g_old)
    print(f'1. flags off: {len(p_new)} parameters, loss {l_new.item():.6f} and every gradient equal the old harness bit for bit')

    p_e = Z.make_params(c, 'polar', 0, embed=True, **BEST)
    assert set(p_e) - set(p_new) == {'embed.harmonic_phasors'} and all(torch.equal(p_e[k], p_new[k]) for k in p_new)
    rows = Z.embed(c, p_e, table, torch.arange(256)).view(256, O.GROUPS, c.width).norm(dim=-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-6)
    l_e, g_e = loss_and_grads(Z, c, p_e, table, w)
    assert not torch.equal(l_e, l_new) and g_e['embed.harmonic_phasors'].abs().sum() > 0
    assert O.role('embed.harmonic_phasors', (256, c.d)) == 'adam'
    print(f'2. --embed on bytes: the old parameters unchanged plus the table (Adam); rows unit a group; loss {l_e.item():.6f} vs {l_new.item():.6f}; the gradient reaches the table')

    V = 2048
    p_t = Z.make_params(c, 'polar', 0, embed=True, vocab=V, **BEST)
    assert p_t['head.weight'].shape == (V, c.features) and p_t['embed.harmonic_phasors'].shape == (V, c.d)
    assert p_t['head.weight'].mean(dim=1).abs().max() < 1e-6
    x = torch.randint(0, V, (1, 64), generator=torch.Generator().manual_seed(1))
    p64, t64 = {k: v.double() for k, v in p_t.items()}, table.double()
    with torch.no_grad():
        chunked = Z.forward(c, p64, t64, x, 'polar')[0]
        s = Z.Sampler(c, p64, t64, 'polar')
        stepped = torch.stack([s.logits(int(t)) for t in x[0]])
    err = (chunked - stepped).abs().max().item()
    assert err < 1e-9, err
    print(f'3. {V} tokens: head {tuple(p_t["head.weight"].shape)} centred; Sampler equals the chunked forward in float64 (max |diff| {err:.1e})')

    class Toy:
        bos = 0
        held_tokens = np.array([5, 9, 1, 7, 7, 300, 2, 2, 2, 11], dtype=np.uint16)
        held_offsets = np.array([0, 3, 4, 10])
        held_bytes = np.array([7, 2, 13])
    which = np.arange(3)
    bpb = field_poles.score_tokens(c, p_t, table, Toy, which, 'polar', 'cpu', batch=2)
    bits = 0.0
    with torch.no_grad():
        for j in which:
            t = torch.from_numpy(Toy.held_tokens[Toy.held_offsets[j]:Toy.held_offsets[j + 1]].astype(np.int64))
            seq = torch.cat([torch.tensor([0]), t])
            # the Sampler, one token at a time: another path than the padded, chunked forward
            s = Z.Sampler(c, p_t, table, 'polar')
            logits = torch.stack([s.logits(int(i)) for i in seq[:-1]])
            bits += F.cross_entropy(logits, seq[1:], reduction='sum').item() / math.log(2)
    hand = bits / Toy.held_bytes.sum()
    assert abs(bpb - hand) < 1e-5, (bpb, hand)
    print(f'4. score_tokens: {bpb:.6f} bpb equals the hand sum {hand:.6f} (3 windows of 3, 1 and 6 tokens, padded to a chunk in batches of 2, against the Sampler)')


if __name__ == '__main__':
    main()
