#!/usr/bin/env python3
"""Export a zara_stt checkpoint for the C inference (zara_stt.c), and dump a Python reference.

    python export_c.py export PARAMS.pt MODEL.bin
    python export_c.py reference PARAMS.pt FILE.wav OUT.f32      # log-probs [frames][classes], prints the greedy text
    python export_c.py random MODEL.bin [--poles 64 --layers 4 --harmonics 32]   # untrained full-size model, for timing

MODEL.bin: "ZSTT", int32 version, front (0 osc, 1 mel), P, nin, D, H, L, dh, layers, classes, W, frame, rate,
then float32 arrays in the order zara_stt.c reads them. Poles are exported as (log r, theta) per step, the
spatial rotation as (cos(dt theta) - 1, sin(dt theta)), gates as sigmoids: the C does no parameterisation.
"""
import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import zara_stt as S  # noqa: E402
import zara_poles as Z  # noqa: E402
import diakosmos_oracle as O  # noqa: E402


def config_of(p):
    D = p['front.probe.weight'].shape[0]
    layers = len({k.split('.')[1] for k in p if k.startswith('blocks.')})
    L = p['blocks.0.temporal.attn.k_harmonic_weight'].shape[1]
    return O.Config(harmonics=D // O.GROUPS // 2, layers=layers, heads=O.GROUPS, kan=L, dt=0.3)


def export(p, out):
    c = config_of(p)
    front = 0 if 'front.pole_decay_log_tau' in p else 1
    nin = p['front.ln.weight'].numel()
    P = nin // 3 if front == 0 else nin
    D, H, L, dh, W = c.d, c.heads, c.kan, c.head_width, c.width
    classes = p['head.weight'].shape[0]
    parts = []

    def put(t):
        parts.append(np.ascontiguousarray(t.detach().to(torch.float32).numpy()).tobytes())
    with open(out, 'wb') as f:
        f.write(b'ZSTT')
        mlp = p['blocks.0.mlp.w1.weight'].shape[0] if 'blocks.0.mlp.w1.weight' in p else 0
        f.write(struct.pack('<14i', 2, front, P, nin, D, H, L, dh, c.layers, classes, W, S.FRAME, S.RATE, mlp))
        if front == 0:
            logr, theta = S.front_pole(p)
            put(logr); put(theta)
        else:
            put(S.mel_matrix('cpu', torch.float32))                 # [MELS][NFFT/2+1]
        put(p['front.ln.weight']); put(p['front.ln.bias']); put(p['front.probe.weight']); put(p['front.probe.bias'])
        for i in range(c.layers):
            b, a = f'blocks.{i}.', f'blocks.{i}.temporal.attn.'
            put(p[b + 'temporal.ln.weight']); put(p[b + 'temporal.ln.bias'])
            put(p[a + 'qkv_proj.weight']); put(p[a + 'qkv_proj.bias'])
            put(p[a + 'q_harmonic_weight']); put(p[a + 'k_harmonic_weight']); put(p[a + 'spectral_coeffs'])
            logr, theta = Z.pole(p, a, 'ct')
            put(logr); put(theta)
            put(p[a + 'out_proj.weight']); put(p[a + 'out_proj.bias'])
            th = c.dt * p[b + 'spatial.theta']
            put(torch.cos(th) - 1); put(torch.sin(th))
            put(p[b + 'spatial.coupling']); put(torch.sigmoid(p[b + 'gate_logit']))
            if mlp:
                m = b + 'mlp.'
                put(p[m + 'ln.weight']); put(p[m + 'ln.bias']); put(p[m + 'w1.weight']); put(p[m + 'w3.weight'])
                put(p[m + 'w2.weight']); put(p[m + 'w2.bias']); put(torch.sigmoid(p[m + 'gate_logit']))
        put(p['ln_f.weight']); put(p['ln_f.bias']); put(p['head.weight']); put(p['head.bias'])
        for b in parts:
            f.write(b)
    print(f'{out}: front {"osc" if front == 0 else "mel"} P {P} D {D} H {H} L {L} dh {dh} layers {c.layers} mlp {mlp} classes {classes}, '
          f'{sum(v.numel() for v in p.values())} params, {Path(out).stat().st_size} bytes')


def load_wav(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype='float32')
    assert sr == S.RATE and x.ndim == 1, (sr, x.shape)
    return x


def reference(p, wav, out):
    c = config_of(p)
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
    q.add_argument('--layers', type=int, default=4); q.add_argument('--harmonics', type=int, default=32); q.add_argument('--front', default='osc')
    q.add_argument('--mlp', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'random':
        c = O.Config(harmonics=a.harmonics, layers=a.layers, heads=7, kan=8, dt=0.3)
        export(S.make_params(c, a.front, a.poles, 0, mlp=a.mlp), a.out)
        return
    p = {k: v.float() for k, v in torch.load(a.params, map_location='cpu').items()}
    if a.cmd == 'export':
        export(p, a.out)
    else:
        reference(p, a.wav, a.out)


if __name__ == '__main__':
    main()
