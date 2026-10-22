#!/usr/bin/env python3
"""Greedy vs CTC beam search with the LibriSpeech 3-gram (OpenSLR 11) on dumped log-probabilities.

    python decode_eval.py DUMP.npz [--lm lm/3-gram.pruned.3e-7.arpa.gz --vocab lm/librispeech-vocab.txt --alpha A --beta B --beam N]
"""
import argparse, gzip, shutil, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import zara_stt as S  # noqa: E402


def wer_cer(hyps, refs):
    W = C = we = ce = 0
    for h, r in zip(hyps, refs):
        we += S.edit_distance(h.split(), r.split()); W += len(r.split())
        ce += S.edit_distance(h, r); C += len(r)
    return we / W, ce / C


ap = argparse.ArgumentParser()
ap.add_argument('dump'); ap.add_argument('--lm', default=''); ap.add_argument('--vocab', default='')
ap.add_argument('--alpha', type=float, nargs='+', default=[0.5]); ap.add_argument('--beta', type=float, nargs='+', default=[1.0])
ap.add_argument('--beam', type=int, default=64)
a = ap.parse_args()
d = np.load(a.dump)
refs = [str(r) for r in d['refs']]
logps = [d[f'logp_{i}'] for i in range(len(refs))]
greedy = [S.greedy(torch.from_numpy(lp), lp.shape[0]) for lp in logps]
print(f'greedy: WER %.3f CER %.3f over {len(refs)} utterances' % wer_cer(greedy, refs))
if a.lm:
    from pyctcdecode import build_ctcdecoder
    lm = a.lm
    if lm.endswith('.gz'):
        if not Path(lm[:-3]).exists():
            with gzip.open(lm, 'rb') as f, open(lm[:-3], 'wb') as g:
                shutil.copyfileobj(f, g)
        lm = lm[:-3]
    labels = [''] + list(S.ALPHABET)
    unigrams = [w.strip() for w in open(a.vocab)] if a.vocab else None
    best = None
    for al in a.alpha:
        for be in a.beta:
            dec = build_ctcdecoder(labels, kenlm_model_path=lm, unigrams=unigrams, alpha=al, beta=be)
            hyp = [dec.decode(lp, beam_width=a.beam) for lp in logps]
            w, c = wer_cer(hyp, refs)
            print(f'beam+LM alpha {al} beta {be} beam {a.beam}: WER {w:.3f} CER {c:.3f}', flush=True)
            if best is None or w < best[0]:
                best = (w, al, be, hyp)
    for i in range(3):
        print('  ref:', refs[i]); print('  greedy:', greedy[i]); print('  LM:', best[3][i])
