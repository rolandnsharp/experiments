#!/usr/bin/env python3
"""Where an update of the pole harness spends its time at Diakosmos's size, and what helps.

    prof_harness.py --bytes DIR --out DIR [--variant polar] [--layers 14 --harmonics 95]
        [--configs 8x8c,16x4c,32x2c,8x8,4x16,8x8c+compile] [--updates 3] [--profile 8x8c]

A config is BATCHxACCUMULATE (64 windows an update either way), `c` for activation
checkpointing per block, `+compile` for torch.compile on the chunked recurrence. Each
config runs one warm-up update and then --updates timed ones (the same windows, so the
numbers compare), each split into forward + backward (all batches) and the optimizer;
peak memory; out-of-memory is recorded, not fatal. --profile names one config whose
first timed update is traced (torch.profiler): the top operations by GPU time.
Writes OUT/profile.json and OUT/profile.txt. The products are the --half ones throughout
(Diakosmos's device numerics), as the run uses.
"""
import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diakosmos_oracle as O  # noqa: E402
import zara_poles  # noqa: E402
from field_poles import TextStream  # noqa: E402
from zara_poles import loss_of, make_params, project  # noqa: E402


def parse(name):
    compile_ = name.endswith('+compile')
    core = name.replace('+compile', '')
    ckpt = core.endswith('c')
    b, k = core.rstrip('c').split('x')
    return int(b), int(k), ckpt, compile_


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def one_update(c, p, table, stream, upd, b, k, ckpt, variant, device, first):
    sync(device)
    t0 = time.time()
    for v in p.values():
        v.requires_grad_(True)
    total = 0.0
    for j in range(k):
        w = torch.from_numpy(stream.windows(first + j * b, b).astype(np.int64)).to(device)
        lj = loss_of(c, p, table, w, variant, checkpoint=ckpt) / k
        lj.backward()
        total += lj.item()
    sync(device)
    t1 = time.time()
    grads = {n: v.grad for n, v in p.items()}
    with torch.no_grad():
        p = {n: v.detach() for n, v in p.items()}
        upd(p, grads)
        project(p, variant)
    sync(device)
    t2 = time.time()
    return p, t1 - t0, t2 - t1, total / math.log(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bytes', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--variant', default='polar')
    ap.add_argument('--layers', type=int, default=14)
    ap.add_argument('--harmonics', type=int, default=95)
    ap.add_argument('--configs', default='8x8c,16x4c,32x2c,8x8,4x16,8x8c+compile')
    ap.add_argument('--updates', type=int, default=3)
    ap.add_argument('--profile', default='8x8c')
    ap.add_argument('--shards', type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zara_poles.HALF = True
    c = O.Config(harmonics=args.harmonics, layers=args.layers, heads=7, kan=8, dt=0.3)
    stream = TextStream(args.bytes, args.shards)
    table = O.wire_table(c.harmonics).to(device)
    log = open(args.out / 'profile.txt', 'w')

    def say(s):
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()
    say(f'{torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}; torch {torch.__version__}; '
        f'{args.variant} {c.layers}L width {c.d}; configs {args.configs}')
    plain = zara_poles.recurrence
    results = []
    for name in args.configs.split(','):
        b, k, ckpt, compile_ = parse(name)
        zara_poles.recurrence = torch.compile(plain, dynamic=False) if compile_ else plain
        r = {'config': name, 'batch': b, 'accumulate': k, 'checkpoint': ckpt, 'compile': compile_}
        try:
            p = {n: v.to(device) for n, v in make_params(c, args.variant, 0).items()}
            upd = O.Update(p, 15360, ns=O.newton_schulz_half)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t = time.time()
            p, f0, o0, l0 = one_update(c, p, table, stream, upd, b, k, ckpt, args.variant, device, 0)
            r['warmupSeconds'] = time.time() - t
            fb, opt = [], []
            for u in range(args.updates):
                if name == args.profile and u == 0 and device.type == 'cuda':
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
                        p, f, o, l = one_update(c, p, table, stream, upd, b, k, ckpt, args.variant, device, 64 * (u + 1))
                    table_txt = prof.key_averages().table(sort_by='cuda_time_total', row_limit=30)
                    say(f'--- the top operations of one {name} update (traced, so slower than untraced)\n{table_txt}')
                    r['tracedSeconds'] = f + o
                    continue
                p, f, o, l = one_update(c, p, table, stream, upd, b, k, ckpt, args.variant, device, 64 * (u + 1))
                fb.append(f)
                opt.append(o)
            r['forwardBackwardSeconds'] = min(fb) if fb else None
            r['optimizerSeconds'] = min(opt) if opt else None
            r['updateSeconds'] = (min(fb) + min(opt)) if fb else None
            r['loss'] = l
            r['peakMemoryGB'] = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None
            say(f'{name:16s} update {r["updateSeconds"]:.2f} s (forward+backward {r["forwardBackwardSeconds"]:.2f}, '
                f'optimizer {r["optimizerSeconds"]:.2f}); warm-up {r["warmupSeconds"]:.1f} s; '
                f'peak {r["peakMemoryGB"] if r["peakMemoryGB"] is None else round(r["peakMemoryGB"], 1)} GB; loss {l:.3f} bpb')
        except torch.cuda.OutOfMemoryError:
            r['error'] = 'out of memory'
            say(f'{name:16s} out of memory')
        except Exception as e:  # noqa: BLE001 - a failing config is a result, not a stop
            r['error'] = f'{type(e).__name__}: {e}'
            say(f'{name:16s} failed: {r["error"][:300]}')
            traceback.print_exc()
        results.append(r)
        p = upd = None
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        (args.out / 'profile.json').write_text(json.dumps({'device': str(device), 'results': results}, indent=1) + '\n')
    zara_poles.recurrence = plain
    ok = [r for r in results if r.get('updateSeconds')]
    if ok:
        best = min(ok, key=lambda r: r['updateSeconds'])
        say(f'fastest: {best["config"]} at {best["updateSeconds"]:.2f} s an update; the remaining 14,720 updates '
            f'{14720 * best["updateSeconds"] / 3600:.1f} h')


if __name__ == '__main__':
    main()
