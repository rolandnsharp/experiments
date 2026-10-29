#!/usr/bin/env python3
"""Export a zara_stt checkpoint for the C inference (zara_stt.c), and dump a Python reference.

    python export_c.py export PARAMS.pt MODEL.bin
    python export_c.py reference PARAMS.pt FILE.wav OUT.f32      # log-probs [frames][classes], prints the greedy text
    python export_c.py random MODEL.bin [--poles 64 --layers 4 --width 448 --states 0 --mlp 0]   # untrained, for timing

MODEL.bin: "ZSTT", int32 version (3), front (0 osc, 1 mel), P, nin, D, N, layers, classes, frame, rate, mlp,
then float32 arrays in the order zara_stt.c reads them. Poles are exported as (log r, theta) per step and
the input gain gamma = sqrt(1 - r^2), the MLP gate as a sigmoid: the C does no parameterisation.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zara_block as K  # noqa: E402
import zara_stt as S  # noqa: E402

VERSION = 3


def shape_of(p):
    D = p['front.probe.weight'].shape[0]
    layers = len({k.split('.')[1] for k in p if k.startswith('blocks.')})
    N = p['blocks.0.pole.log_tau'].numel()
    return S.Shape(D, layers, N)


def export(p, out):
    c = shape_of(p)
    front = 0 if 'front.pole_decay_log_tau' in p else 1
    nin = p['front.ln.weight'].numel()
    P = nin // 3 if front == 0 else nin
    D, N = c.d, c.states
    classes = p['head.weight'].shape[0]
    mlp = p['blocks.0.mlp.w1.weight'].shape[0] if 'blocks.0.mlp.w1.weight' in p else 0
    parts = []

    def put(t):
        parts.append(np.ascontiguousarray(t.detach().to(torch.float32).numpy()).tobytes())
    if front == 0:
        logr, theta = S.front_pole(p)
        put(logr); put(theta)
    else:
        put(S.mel_matrix('cpu', torch.float32))                     # [MELS][NFFT/2+1]
    put(p['front.ln.weight']); put(p['front.ln.bias']); put(p['front.probe.weight']); put(p['front.probe.bias'])
    for i in range(c.layers):
        b = f'blocks.{i}.'
        put(p[b + 'ln.weight']); put(p[b + 'ln.bias'])
        logr, theta = K.pole(p, b)
        put(logr); put(theta); put(torch.sqrt(-torch.expm1(2 * logr)))
        put(p[b + 'in.weight']); put(p[b + 'out.weight']); put(p[b + 'skip'])
        put(p[b + 'gate.weight']); put(p[b + 'proj.weight']); put(p[b + 'proj.bias'])
        if mlp:
            m = b + 'mlp.'
            put(p[m + 'ln.weight']); put(p[m + 'ln.bias']); put(p[m + 'w1.weight']); put(p[m + 'w3.weight'])
            put(p[m + 'w2.weight']); put(p[m + 'w2.bias']); put(torch.sigmoid(p[m + 'gate_logit']))
    put(p['ln_f.weight']); put(p['ln_f.bias']); put(p['head.weight']); put(p['head.bias'])
    with open(out, 'wb') as f:
        f.write(b'ZSTT')
        f.write(struct.pack('<11i', VERSION, front, P, nin, D, N, c.layers, classes, S.FRAME, S.RATE, mlp))
        for part in parts:
            f.write(part)
    print(f'{out}: front {"osc" if front == 0 else "mel"} P {P} D {D} N {N} layers {c.layers} mlp {mlp} classes {classes}, '
          f'{sum(v.numel() for v in p.values())} params, {Path(out).stat().st_size} bytes')


def load_wav(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype='float32')
    assert sr == S.RATE and x.ndim == 1, (sr, x.shape)
    return x


def reference(p, wav, out):
    c = shape_of(p)
    front = 'osc' if 'front.pole_decay_log_tau' in p else 'mel'
    x = load_wav(wav)
    x = x / (np.sqrt((x ** 2).mean()) + 1e-5) * 0.1
    T = -(-x.size // S.FRAME) * S.FRAME
    xp = np.zeros(T, dtype=np.float32)
    xp[:x.size] = x
    with torch.no_grad():
        logp = S.forward(c, p, torch.from_numpy(xp)[None], front)[0]
    logp.numpy().astype(np.float32).tofile(out)
    print(f'{out}: {tuple(logp.shape)}')
    print('greedy:', S.greedy(logp, logp.shape[0]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('export'); e.add_argument('params'); e.add_argument('out')
    r = sub.add_parser('reference'); r.add_argument('params'); r.add_argument('wav'); r.add_argument('out')
    q = sub.add_parser('random'); q.add_argument('out'); q.add_argument('--poles', type=int, default=64)
    q.add_argument('--layers', type=int, default=4); q.add_argument('--width', type=int, default=448)
    q.add_argument('--states', type=int, default=0); q.add_argument('--front', default='osc'); q.add_argument('--mlp', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'random':
        export(S.make_params(S.Shape(a.width, a.layers, a.states), a.front, a.poles, 0, mlp=a.mlp), a.out)
        return
    p = {k: v.float() for k, v in torch.load(a.params, map_location='cpu').items()}
    if a.cmd == 'export':
        export(p, a.out)
    else:
        reference(p, a.wav, a.out)


if __name__ == '__main__':
    main()
