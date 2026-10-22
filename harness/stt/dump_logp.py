#!/usr/bin/env python3
"""Dump a checkpoint's CTC log-probabilities for the first N utterances of a corpus (CPU).

    python dump_logp.py PARAMS.pt DATA NAME N OUT.npz

OUT.npz: logp_<i> [frames][classes] float32, refs (list of texts), ids. For decoder experiments.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zara_stt as S  # noqa: E402
import diakosmos_oracle as O  # noqa: E402

params, data, name, n, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
p = {k: v.float() for k, v in torch.load(params, map_location='cpu').items()}
D = p['front.probe.weight'].shape[0]
layers = len({k.split('.')[1] for k in p if k.startswith('blocks.')})
c = O.Config(harmonics=D // 14, layers=layers, heads=7, kan=p['blocks.0.temporal.attn.k_harmonic_weight'].shape[1], dt=0.3)
front = 'osc' if 'front.pole_decay_log_tau' in p else 'mel'
corpus = S.Corpus(data, name, 40.0)
items = corpus.items[:n]
arrays, refs, ids = {}, [], []
with torch.no_grad():
    for i in range(0, len(items), 8):
        batch = items[i:i + 8]
        x, lengths, _, _ = corpus.batch(batch, torch.device('cpu'))
        logp = S.forward(c, p, x, front)
        for k, u in enumerate(batch):
            arrays[f'logp_{i + k}'] = logp[k, :int(lengths[k])].numpy().astype(np.float32)
            refs.append(u['text'].upper()); ids.append(u['id'])
        print(f'{i + len(batch)}/{len(items)}', flush=True)
np.savez(out, refs=np.array(refs), ids=np.array(ids), **arrays)
print(out, len(items), 'utterances')
