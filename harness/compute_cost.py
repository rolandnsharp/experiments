#!/usr/bin/env python3
"""Count the multiply-adds a pole-recurrence byte model spends, per byte and over a
training run, split into matrix products and the recurrence itself, beside GPT-2.

    compute_cost.py RUN_DIR [RUN_DIR ...]      reads each result.json, writes compute.json beside it
    compute_cost.py --config L H [dense|grouped]   a hypothetical run: L layers, H harmonics

Counts are multiply-accumulates (MAC) per byte of context for one forward pass;
a training step costs about three forwards (forward, and the backward's two
products). GPT-2 124M is counted per BPE token and divided by 4.4 bytes a token
(OpenWebText average), attention included at its 1024 context.
"""
import json
import sys
from pathlib import Path

GROUPS, HEADS, L, CHUNK, VOCAB = 7, 7, 8, 64, 256


def per_byte(layers, harmonics, variant, mixing='dense'):
    width = 2 * harmonics
    d = GROUPS * width
    dh = width
    complex_pole = variant != 'real'
    fan = d if mixing == 'dense' else dh        # grouped: each head projects its own dh channels
    qkv = 3 * d * fan
    harmonic = 2 * d * L                       # q and k projections onto L harmonics a head
    out = d * fan
    # recurrence: within-chunk kernel over the chunk's C positions, the read, the carried state
    kernel = HEADS * L * CHUNK * (2 if complex_pole else 1)
    read = CHUNK * d
    carry = (4 + (2 if complex_pole else 1) + 2) * L * d
    spatial = GROUPS * width * 4
    block = {'matrix products': qkv + harmonic + out, 'recurrence': kernel + read + carry, 'spatial rotation': spatial}
    head = (d + 2 * GROUPS) * VOCAB + 3 * d
    total = {k: v * layers for k, v in block.items()}
    total['head'] = head
    total['total'] = sum(total.values())
    return d, total


def gpt2_per_byte():
    d, layers, T, vocab, bpt = 768, 12, 1024, 50257, 4.4
    block = 12 * d * d + 2 * T * d            # projections and MLP, then QK and AV over the context
    per_token = layers * block + d * vocab
    return per_token, per_token / bpt


def report(name, layers, harmonics, variant, updates=None, batch=None, context=2048, seconds=None, mixing='dense'):
    d, cost = per_byte(layers, harmonics, variant, mixing)
    matrix = cost['matrix products'] + cost['head']
    out = {'model': name, 'layers': layers, 'harmonics': harmonics, 'width': d, 'variant': variant, 'mixing': mixing,
           'macPerByte': cost, 'matrixShare': matrix / cost['total']}
    if updates:
        seen = updates * batch * context
        out['training'] = {'bytesSeen': seen, 'mac': 3 * cost['total'] * seen, 'flop': 6 * cost['total'] * seen}
        if seconds:
            out['training']['seconds'] = seconds
    tok, byte = gpt2_per_byte()
    out['gpt2_124m'] = {'macPerToken': tok, 'macPerByte': byte, 'ratioToThis': byte / cost['total'],
                        'trainingFlopOn9BTokens': 6 * 124e6 * 9.03e9}
    return out


def main():
    if sys.argv[1] == '--config':
        layers, harmonics = int(sys.argv[2]), int(sys.argv[3])
        mixing = sys.argv[4] if len(sys.argv) > 4 else 'dense'
        print(json.dumps(report('hypothetical', layers, harmonics, 'polar', mixing=mixing), indent=1))
        return
    for run in map(Path, sys.argv[1:]):
        r = json.loads((run / 'result.json').read_text())
        out = report(run.name, r['layers'], r['harmonics'], r['variant'], r.get('updates'), r.get('batch'), seconds=r.get('seconds'), mixing=r.get('mixing', 'dense'))
        (run / 'compute.json').write_text(json.dumps(out, indent=1) + '\n')
        c = out['macPerByte']
        print(f"{run.name}: {c['total']/1e6:.2f} M MAC/byte (matrix {out['matrixShare']*100:.0f}%, recurrence {c['recurrence']/c['total']*100:.0f}%), "
              f"GPT-2 124M = {out['gpt2_124m']['ratioToThis']:.0f}x that per byte" +
              (f"; training {out['training']['flop']:.2e} FLOP over {out['training']['bytesSeen']/1e6:.0f} MB" if 'training' in out else ''))


if __name__ == '__main__':
    main()
