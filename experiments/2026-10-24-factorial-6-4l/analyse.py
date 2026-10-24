#!/usr/bin/env python3
"""EXP-0029's pre-registered analysis: a 2^(5-1) half fraction (I = -ABCDE) in mlp, select, phase,
conv, gate. Reads each design point's final held-out bpb from the run logs on the lab copy, fits
bpb = b0 + sum b_i x_i + sum b_ij x_i x_j (x = -1/+1, saturated on 16 points), prints the effects
(2b), Lenth's pseudo standard error, the seed-replicate noise, and the fitted best of all 32.
Usage: analyse.py RUNS_DIR"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np

runs = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs')
F = 'mspcg'
NAMES = dict(m='mlp', s='select', p='phase', c='conv', g='gate')
# the single-factor points are the screens' runs; the select screen was rerun on the fixed code (sel2)
SINGLE = dict(m='polar-dense-4L20H-s{s}-u3000-b16-mlp600', s='polar-dense-4L32H-s{s}-u3000-b16-sel2',
              p='polar-dense-4L32H-s{s}-u3000-b16-phase', c='polar-dense-4L32H-s{s}-u3000-b16-conv4',
              g='polar-dense-4L32H-s{s}-u3000-b16-gate16')
CONTROL = dict(s0=2.0490, s1=2.0409)


def final(name):
    f = runs / name / 'train.log'
    if not f.exists():
        return None
    m = re.findall(r'final held-out \(9052 of 9052.*?\) ([\d.]+) bpb', f.read_text())
    return float(m[-1]) if m else None


def point(code, seed=0):
    if len(code) == 1:
        return final(SINGLE[code].format(s=seed))
    h = 20 if 'm' in code else 32
    return final(f'polar-dense-4L{h}H-s{seed}-u3000-b16-fx-{code}')


design = [''.join(c) for r in (1, 3, 5) for c in itertools.combinations(F, r)]
y = {code: point(code) for code in design}
missing = [c for c, v in y.items() if v is None]
for c in design:
    print(f'{c:6s} {y[c] if y[c] is not None else "-"}')
if missing:
    print(f'\n{len(missing)} of 16 points missing: {" ".join(missing)}')
    sys.exit(0)

terms = [(i,) for i in range(5)] + list(itertools.combinations(range(5), 2))
label = lambda t: ' x '.join(NAMES[F[i]] for i in t)
x = lambda code: np.array([1 if f in code else -1 for f in F])
X = np.array([[1.0] + [np.prod(x(c)[list(t)]) for t in terms] for c in design])
b = np.linalg.solve(X, np.array([y[c] for c in design]))
effects = {label(t): 2 * b[k + 1] for k, t in enumerate(terms)}
a = np.abs(list(effects.values()))
s0 = 1.5 * np.median(a)
pse = 1.5 * np.median(a[a < 2.5 * s0])
reps = [(point(c, 0), point(c, 1)) for c in 'mspcg'] + [(CONTROL['s0'], CONTROL['s1'])]
reps = [(u, v) for u, v in reps if u is not None and v is not None]
sd = np.sqrt(np.mean([(u - v) ** 2 / 2 for u, v in reps])) if reps else float('nan')
se_eff = 2 * sd / np.sqrt(16)
print(f'\nintercept {b[0]:.4f} (fitted mean over the half fraction)')
print(f"Lenth PSE {pse:.4f}; seed sd {sd:.4f} from {len(reps)} replicate pairs -> effect se {se_eff:.4f}")
print('\neffect (2b), most negative (most helpful) first:')
for k, v in sorted(effects.items(), key=lambda kv: kv[1]):
    flag = '*' if abs(v) >= 0.005 and abs(v) > 2 * max(pse, se_eff) else ' '
    print(f'  {flag} {k:20s} {v:+.4f}')
pred = {}
for r in range(6):
    for c in itertools.combinations(F, r):
        code = ''.join(c)
        pred[code] = float(np.array([1.0] + [np.prod(x(code)[list(t)]) for t in terms]) @ b)
best = sorted(pred.items(), key=lambda kv: kv[1])[:5]
print('\nfitted best of all 32 (control "" fitted', f"{pred['']:.4f}, measured {np.mean(list(CONTROL.values())):.4f}):")
for code, v in best:
    print(f"  {'+'.join(NAMES[f] for f in code) or 'none':35s} {v:.4f}")
