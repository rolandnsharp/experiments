#!/usr/bin/env python3
"""ARPA 3-gram -> the compact binary LM read by zara_stt.c (--lm).

    python export_lm.py LM.arpa OUT.bin

Format "ZLM1": int32 V, N2, N3; then V words (uint16 length + bytes) in sorted byte order (the
word id is the index); float32 uni_logp[V], uni_backoff[V]; uint64 bigram keys[N2] (sorted,
w1 << 18 | w2), float32 bi_logp[N2], bi_backoff[N2]; uint64 trigram keys[N3] (sorted,
w1 << 36 | w2 << 18 | w3), float32 tri_logp[N3]. Log-probabilities are natural logs (ARPA's
log10 times ln 10). <s>, </s> and <unk> are ordinary words in the table.
"""
import math, struct, sys
import numpy as np

src, out = sys.argv[1], sys.argv[2]
LN10 = math.log(10)
grams = {1: [], 2: [], 3: []}
order = 0
with open(src, encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('\\') and line.endswith('-grams:'):
            order = int(line[1]); continue
        if not line or line.startswith('\\') or order == 0 or line.startswith('ngram '):
            continue
        parts = line.split('\t')
        lp = float(parts[0]); words = parts[1].split(' '); bo = float(parts[2]) if len(parts) > 2 else 0.0
        grams[order].append((words, lp, bo))
vocab = sorted({w[0][0] for w in grams[1]}, key=lambda s: s.encode())
assert len(vocab) < (1 << 18)
wid = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
uni_lp = np.full(V, -99.0, np.float32); uni_bo = np.zeros(V, np.float32)
for (w,), lp, bo in ((g[0], g[1], g[2]) for g in grams[1]):
    uni_lp[wid[w]] = lp * LN10; uni_bo[wid[w]] = bo * LN10
def table(n):
    rows = []
    for words, lp, bo in grams[n]:
        k = 0
        for w in words:
            k = (k << 18) | wid[w]
        rows.append((k, lp * LN10, bo * LN10))
    rows.sort()
    return (np.array([r[0] for r in rows], np.uint64), np.array([r[1] for r in rows], np.float32),
            np.array([r[2] for r in rows], np.float32))
k2, l2, b2 = table(2)
k3, l3, _ = table(3)
with open(out, 'wb') as f:
    f.write(b'ZLM1'); f.write(struct.pack('<3i', V, len(k2), len(k3)))
    for w in vocab:
        b = w.encode(); f.write(struct.pack('<H', len(b))); f.write(b)
    for a in (uni_lp, uni_bo, k2, l2, b2, k3, l3):
        f.write(a.tobytes())
print(f'{out}: {V} words, {len(k2)} bigrams, {len(k3)} trigrams')
