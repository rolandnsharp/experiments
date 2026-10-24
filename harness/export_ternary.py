#!/usr/bin/env python3
"""Export a field_poles.py checkpoint for the C forward (vidya_ternary.c), and dump a PyTorch reference.

    python export_ternary.py export RUN_DIR MODEL.bin [--ternary] [--act8] [--int8]
    python export_ternary.py reference RUN_DIR REF.bin [--bytes 2048] [--ternary] [--act8]

The forward is result.json's ('ternary', 'act8'); the flags force them for runs whose result.json
predates those keys. --int8 (a float run only) quantizes the projections after training: int8 weights
with one absmax scale a row, the usual post-training scheme, for comparison with ternary training.

MODEL.bin: b"VTRN", then int32 version, D, H, L, dh, layers, G, W, features, kind (0 float, 1 ternary,
2 int8), act8, then per
block: temporal.ln weight and bias; the qkv projection; q_harmonic_weight [H][2L][dh], k_harmonic_weight
[H][L][dh], spectral_coeffs [H][L]; the pole as (Re c, Im c) [H][L], computed as Sampler computes it
(exp(log r) cos theta, exp(log r) sin theta in float32); the out projection; the spatial rotation as
(cos(dt theta) - 1, sin(dt theta)) [G/2][W]; spatial.coupling [G]; sigmoid(gate_logit) [G]. Then ln_f
weight and bias, head.weight [256][features], and the wire table [256][D]. A projection [rows][cols] is
float32 weights then float32 bias when ternary = 0; when ternary = 1 it is the float32 absmean scale, the
codes packed four to a byte (2 bits each, low bits first: 0 -> 0, 1 -> +1, 2 -> -1), then the float32 bias;
when kind = 2 it is float32 scales [rows] (max |row| / 127), int8 weights [rows][cols], then the bias.
Everything little-endian. The C does no parameterisation.

REF.bin: int32 n, the n held-out bytes (the first n of the GPT-2 baseline split), then Sampler's float32
logits [n][256] after each byte. The script prints the next-byte loss over the n - 1 predictions in bits.
"""
import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diakosmos_oracle as O  # noqa: E402
import zara_poles as Z  # noqa: E402

VERSION = 1


def load(run, ternary, act8):
    meta = json.loads((run / 'result.json').read_text())
    if meta['variant'] != 'polar' or meta.get('mixing', 'dense') != 'dense':
        raise SystemExit('the C forward is the polar pole with dense projections')
    c = O.Config(harmonics=meta['harmonics'], layers=meta['layers'], heads=7, kan=8, dt=0.3)
    p = torch.load(run / 'params.pt', map_location='cpu')
    Z.TERNARY = ternary or meta.get('ternary', False)
    Z.ACT8 = act8 or meta.get('act8', False)
    return c, p


def export(args):
    c, p = load(args.run, args.ternary, args.act8)
    if args.int8 and Z.TERNARY:
        raise SystemExit('--int8 quantizes a float run after training, not a ternary one')
    D, H, L, dh, W, G = c.d, c.heads, c.kan, c.head_width, c.width, O.GROUPS
    parts = []

    def put(t):
        parts.append(np.ascontiguousarray(t.detach().to(torch.float32).numpy()).tobytes())

    def projection(name):
        w = p[name]
        if args.int8:
            s = (w.abs().amax(dim=1) / 127).clamp(min=1e-12)
            put(s)
            parts.append((w / s[:, None]).round().clamp(-127, 127).to(torch.int8).numpy().tobytes())
            return
        if not Z.TERNARY:
            put(w)
            return
        s = w.abs().mean().clamp(min=1e-5)
        q = (w / s).round().clamp(-1, 1).to(torch.int64).flatten()
        # the forward's w + (q s - w) equals q s up to one rounding
        assert torch.allclose(Z.ternary(w), q.view_as(w).to(w.dtype) * s, rtol=0, atol=1e-6 * float(s))
        code = torch.where(q < 0, 2, q).numpy().astype(np.uint8)
        code = np.concatenate([code, np.zeros((-len(code)) % 4, np.uint8)]).reshape(-1, 4)
        put(s.reshape(1))
        parts.append((code[:, 0] | code[:, 1] << 2 | code[:, 2] << 4 | code[:, 3] << 6).astype(np.uint8).tobytes())

    for i in range(c.layers):
        b, a = f'blocks.{i}.', f'blocks.{i}.temporal.attn.'
        put(p[b + 'temporal.ln.weight']); put(p[b + 'temporal.ln.bias'])
        projection(a + 'qkv_proj.weight'); put(p[a + 'qkv_proj.bias'])
        put(p[a + 'q_harmonic_weight']); put(p[a + 'k_harmonic_weight']); put(p[a + 'spectral_coeffs'])
        logr, theta = Z.pole(p, a, 'polar')
        put(torch.exp(logr) * torch.cos(theta)); put(torch.exp(logr) * torch.sin(theta))
        projection(a + 'out_proj.weight'); put(p[a + 'out_proj.bias'])
        th = c.dt * p[b + 'spatial.theta']
        put(torch.cos(th) - 1); put(torch.sin(th))
        put(p[b + 'spatial.coupling']); put(torch.sigmoid(p[b + 'gate_logit']))
    put(p['ln_f.weight']); put(p['ln_f.bias']); put(p['head.weight']); put(O.wire_table(c.harmonics))
    with open(args.out, 'wb') as f:
        f.write(b'VTRN')
        kind = 2 if args.int8 else int(Z.TERNARY)
        f.write(struct.pack('<11i', VERSION, D, H, L, dh, c.layers, G, W, c.features, kind, int(Z.ACT8)))
        for part in parts:
            f.write(part)
    size = Path(args.out).stat().st_size
    print(f'{args.out}: {size} bytes ({size / 1e6:.2f} MB), {c.layers}L D={D} H={H} L={L} dh={dh}, kind {kind} act8 {int(Z.ACT8)}')


def reference(args):
    c, p = load(args.run, args.ternary, args.act8)
    held = np.fromfile(Path(__file__).resolve().parent / 'owt' / 'heldout.bytes', dtype=np.uint8, count=args.bytes)
    if held.size < args.bytes:
        raise SystemExit('owt/heldout.bytes is shorter than --bytes')
    table = O.wire_table(c.harmonics)
    s = Z.Sampler(c, p, table, 'polar')
    logits = torch.stack([s.logits(int(b)) for b in held])
    nats = F.cross_entropy(logits[:-1], torch.from_numpy(held[1:].astype(np.int64)))
    with open(args.out, 'wb') as f:
        f.write(struct.pack('<i', held.size))
        f.write(held.tobytes())
        f.write(logits.to(torch.float32).numpy().tobytes())
    print(f'{args.out}: {held.size} bytes; Sampler next-byte loss {nats.item() / math.log(2):.6f} bits over {held.size - 1} predictions')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('export', export), ('reference', reference)):
        s = sub.add_parser(name)
        s.add_argument('run', type=Path)
        s.add_argument('out', type=Path)
        s.add_argument('--ternary', action='store_true')
        s.add_argument('--act8', action='store_true')
        s.add_argument('--int8', action='store_true')
        if name == 'reference':
            s.add_argument('--bytes', type=int, default=2048)
        s.set_defaults(fn=fn)
    args = ap.parse_args()
    with torch.no_grad():
        args.fn(args)


if __name__ == '__main__':
    main()
