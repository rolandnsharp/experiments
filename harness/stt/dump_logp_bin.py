#!/usr/bin/env python3
"""dump_logp.py's .npz -> a flat file zara_stt.c --decode-logp reads: int32 count, then per
utterance int32 frames and float32 [frames][29]. Also writes the references, one per line."""
import struct, sys
import numpy as np
d = np.load(sys.argv[1]); refs = [str(r) for r in d['refs']]
with open(sys.argv[2], 'wb') as f:
    f.write(struct.pack('<i', len(refs)))
    for i in range(len(refs)):
        a = d[f'logp_{i}'].astype(np.float32); f.write(struct.pack('<i', a.shape[0])); f.write(a.tobytes())
open(sys.argv[3], 'w').write('\n'.join(refs) + '\n')
print(len(refs), 'utterances')
