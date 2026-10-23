#!/usr/bin/env python3
"""The pole recurrence on text bytes: real decay vs Cartesian vs polar pole, scored
on Diakosmos's held-out OpenWebText bytes.

    field_poles.py train --variant real|cartesian|polar --bytes DIR --out DIR
        [--layers 4 --harmonics 32 --batch 16 --updates 3000 --eval-every 500 --shards N]
        [--accumulate K] [--half] [--stop-at N]

--accumulate K: an update is the mean gradient of K batches of --batch windows (the
windows read in order), so --batch 16 --accumulate 4 is Diakosmos's 64 windows an
update. --half: the matrix products and Muon's Newton-Schulz as Diakosmos's device
computes them (bfloat16 operands, binary32 accumulation; zara_poles.HALF and the
oracle's newton_schulz_half). --stop-at N: after update N of the run's schedule (still
the schedule for --updates), save the checkpoint, score the full held-out set
(--final-windows) and stop, writing stopped-N.json; --resume continues the run. Without
these flags the run is as before, bit for bit.

The model, the optimizer and the scoring are zara_poles.py's (the audio
harness, itself Diakosmos's oracle with the recurrence replaced); only the data
and the continuation differ: windows come from OpenWebText byte shards
(prepare_owt_bytes.py) and continuations are text sampled after held-out
prompts, written to result.json and samples.txt.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diakosmos_oracle as O  # noqa: E402
import zara_poles  # noqa: E402
from zara_poles import Sampler, loss_of, make_params, pole, project, score  # noqa: E402


class ByteShards:
    """The training shards in manifest order as one byte stream, memory-mapped:
    the same bytes np.concatenate would give, without holding them all."""
    def __init__(self, d, shards):
        self.maps = [np.memmap(d / s['train']['file'], dtype=np.uint8, mode='r') for s in shards]
        self.starts = np.cumsum([0] + [m.size for m in self.maps])
        self.size = int(self.starts[-1])

    def __getitem__(self, span):
        a, b = span.start, span.stop
        parts = []
        i = int(np.searchsorted(self.starts, a, side='right') - 1)
        while a < b:
            m = self.maps[i]
            lo, hi = a - int(self.starts[i]), min(b - int(self.starts[i]), m.size)
            parts.append(np.asarray(m[lo:hi]))
            a += hi - lo
            i += 1
        return parts[0] if len(parts) == 1 else np.concatenate(parts)


class TextStream:
    def __init__(self, d, shards=0):
        m = json.loads((d / 'manifest.json').read_text())
        self.m = m
        used = m['shards'] if shards == 0 else m['shards'][:shards]
        if shards and len(used) < shards:
            raise SystemExit(f'{shards} shards asked for, {len(used)} in the manifest')
        self.shards = len(used)
        self.cat = ByteShards(d, used)
        h = m['heldout']
        text = np.fromfile(d / h['file'], dtype=np.uint8)
        count = -(-h['scoredBytes'] // O.FULL_CONTEXT)
        stream = np.zeros(count * O.FULL_CONTEXT + 1, dtype=np.uint8)
        stream[:text.size - h['contextBytes'] + 1] = text[h['contextBytes'] - 1:]
        self.heldout = np.lib.stride_tricks.as_strided(stream, shape=(count, O.FULL_CONTEXT + 1), strides=(O.FULL_CONTEXT, 1)).copy()
        self.scored = h['scoredBytes']
        self.text = text

    def windows(self, first, count):
        T = O.FULL_CONTEXT
        end = (first + count) * T + 1
        if end > self.cat.size:
            raise SystemExit(f'the training stream ({self.cat.size} bytes) ends before window {first + count}')
        return np.stack([self.cat[k * T:k * T + T + 1] for k in range(first, first + count)])

    def prompts(self, n=3, length=256):
        span = self.text.size - 2 * length
        return [self.text[(i + 1) * span // (n + 1):(i + 1) * span // (n + 1) + length] for i in range(n)]


def show(b):
    return bytes(b).decode('utf-8', errors='replace')


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zara_poles.HALF = args.half
    c = O.Config(harmonics=args.harmonics, layers=args.layers, heads=7, kan=8, dt=0.3)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'train.log', 'a')

    def say(s):
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()
    stream = TextStream(args.bytes, args.shards)
    table = O.wire_table(c.harmonics).to(device)
    p = {k: v.to(device) for k, v in make_params(c, args.variant, args.seed, mixing=args.mixing).items()}
    count = sum(v.numel() for v in p.values())
    say(f'variant {args.variant} mixing {args.mixing} layers {c.layers} width {c.d} params {count} device {device} batch {args.batch} updates {args.updates} '
        f'shards {stream.shards} train bytes {stream.cat.size} held-out windows {stream.heldout.shape[0]} scored bytes {stream.scored}'
        + (f' accumulate {args.accumulate} ({args.batch * args.accumulate} windows an update)' if args.accumulate > 1 else '')
        + (' half (bfloat16 product operands, binary32 accumulation)' if args.half else ''))
    upd = O.Update(p, args.updates, ns=O.newton_schulz_half if args.half else None)
    per = args.batch * args.accumulate
    eval_windows = stream.heldout[:args.eval_windows]
    t0 = time.time()
    curve = []
    first = 0
    if args.resume and (out / 'resume.pt').exists():
        # the run's state at its latest held-out check: parameters, optimizer, step, curve, clock
        r = torch.load(out / 'resume.pt', map_location=device)
        p = {k: v.to(device) for k, v in r['params'].items()}
        upd.state = {k: {n: (v.to(device) if torch.is_tensor(v) else v) for n, v in st.items()} for k, st in r['state'].items()}
        upd.step = r['step']
        first = r['step']
        curve = r['curve']
        t0 = time.time() - r['seconds']
        say(f'resumed at step {first} ({r["seconds"]:.0f} s so far, held-out {curve[-1]["bpb"]:.4f} bpb)')
    for step in range(first, args.updates):
        for v in p.values():
            v.requires_grad_(True)
        if args.accumulate == 1:
            w = torch.from_numpy(stream.windows(step * args.batch, args.batch).astype(np.int64)).to(device)
            l = loss_of(c, p, table, w, args.variant, checkpoint=args.checkpoint)
            l.backward()
        else:
            # the mean over the update's windows: each batch's mean loss over K, summed
            total = 0.0
            for j in range(args.accumulate):
                w = torch.from_numpy(stream.windows(step * per + j * args.batch, args.batch).astype(np.int64)).to(device)
                lj = loss_of(c, p, table, w, args.variant, checkpoint=args.checkpoint) / args.accumulate
                lj.backward()
                total += lj.item()
            l = torch.tensor(total)
        grads = {k: v.grad for k, v in p.items()}
        with torch.no_grad():
            p = {k: v.detach() for k, v in p.items()}
            norm = upd(p, grads)
            project(p, args.variant)
        if step % 20 == 0 or step == args.updates - 1:
            say(f'step {step:6d}  loss {l.item() / math.log(2):.4f} bpb  gradnorm {norm:.3f}  {time.time() - t0:.0f} s')
        if (step + 1) % args.eval_every == 0 or step == args.updates - 1:
            bpb = score(c, p, table, eval_windows, args.variant, device)
            curve.append({'step': step + 1, 'bpb': bpb, 'seconds': time.time() - t0})
            say(f'step {step:6d}  held-out ({eval_windows.shape[0]} windows) {bpb:.4f} bpb')
            # the checkpoint so far, so a run that dies leaves its latest state
            torch.save({k: v.cpu() for k, v in p.items()}, out / 'params.pt')
            (out / 'checkpoint.json').write_text(json.dumps({'step': step + 1, 'bpb': bpb, 'curve': curve}) + '\n')
            torch.save({'params': {k: v.cpu() for k, v in p.items()},
                        'state': {k: {n: (v.cpu() if torch.is_tensor(v) else v) for n, v in st.items()} for k, st in upd.state.items()},
                        'step': upd.step, 'curve': curve, 'seconds': time.time() - t0}, out / 'resume.pt.tmp')
            os.replace(out / 'resume.pt.tmp', out / 'resume.pt')
        if args.stop_at and step + 1 == args.stop_at:
            if (step + 1) % args.eval_every != 0 and step != args.updates - 1:
                raise SystemExit('--stop-at must fall on a held-out check (a multiple of --eval-every)')
            final = stream.heldout if args.final_windows == 0 else stream.heldout[:args.final_windows]
            full = score(c, p, table, final, args.variant, device)
            say(f'stopped after update {step + 1} of {args.updates}: held-out ({final.shape[0]} of {stream.heldout.shape[0]} windows) {full:.4f} bpb')
            (out / f'stopped-{step + 1}.json').write_text(json.dumps({'step': step + 1, 'updates': args.updates, 'heldoutBpb': full,
                                                                      'windows': int(final.shape[0]), 'seconds': time.time() - t0,
                                                                      'params': count, 'curve': curve}) + '\n')
            return
    torch.save({k: v.cpu() for k, v in p.items()}, out / 'params.pt')
    final = stream.heldout if args.final_windows == 0 else stream.heldout[:args.final_windows]
    full = score(c, p, table, final, args.variant, device)
    say(f'final held-out ({final.shape[0]} of {stream.heldout.shape[0]} windows, {stream.scored} scored bytes in all) {full:.4f} bpb')
    result = {'variant': args.variant, 'mixing': args.mixing, 'layers': c.layers, 'harmonics': c.harmonics, 'params': count,
              'updates': args.updates, 'batch': args.batch, 'accumulate': args.accumulate, 'half': args.half,
              'trainBytesSeen': args.updates * per * O.FULL_CONTEXT,
              'shards': stream.shards, 'trainBytes': stream.cat.size,
              'seconds': time.time() - t0, 'heldoutBpb': full, 'curve': curve, 'samples': {}}
    logr, theta = pole(p, 'blocks.0.temporal.attn.', args.variant)
    result['block0Poles'] = {'periodBytes': (2 * math.pi / theta.abs().clamp(min=1e-9)).flatten().tolist(),
                             'decayBytes': (-1 / logr).flatten().tolist()}
    gen = torch.Generator(device=device).manual_seed(args.seed)
    with open(out / 'samples.txt', 'w') as f:
        for temperature in (1.0, 0.8):
            rows = []
            for i, prompt in enumerate(stream.prompts()):
                s = Sampler(c, p, table, args.variant)
                g = s.run(prompt, 256, temperature, gen)
                rows.append({'prompt': show(prompt), 'generated': show(g)})
                f.write(f'=== T={temperature} prompt {i}\n{show(prompt)}\n--- continuation\n{show(g)}\n\n')
            result['samples'][str(temperature)] = rows
    say('sample (T=1.0): ' + result['samples']['1.0'][0]['generated'][:200].replace('\n', ' '))
    (out / 'result.json').write_text(json.dumps(result, indent=1) + '\n')
    say('done')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--variant', required=True, choices=['real', 'cartesian', 'polar'])
    t.add_argument('--bytes', type=Path, required=True)
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--mixing', default='dense', choices=['dense', 'grouped'], help='dense: Diakosmos projections over all heads; grouped: each head projects its own channels')
    t.add_argument('--layers', type=int, default=4)
    t.add_argument('--harmonics', type=int, default=32)
    t.add_argument('--batch', type=int, default=16)
    t.add_argument('--updates', type=int, default=3000)
    t.add_argument('--eval-every', type=int, default=500)
    t.add_argument('--eval-windows', type=int, default=256)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--checkpoint', action='store_true')
    t.add_argument('--final-windows', type=int, default=0, help='score this many held-out windows at the end (0: all)')
    t.add_argument('--shards', type=int, default=0, help='train on the first N shards of the manifest (0: all)')
    t.add_argument('--resume', action='store_true', help='continue from OUT/resume.pt when it exists')
    t.add_argument('--accumulate', type=int, default=1, help='batches of --batch windows an update (default 1)')
    t.add_argument('--half', action='store_true', help="the products and Newton-Schulz as Diakosmos's device computes them")
    t.add_argument('--stop-at', type=int, default=0, help='stop after this update (a held-out check), scoring the full held-out set')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
