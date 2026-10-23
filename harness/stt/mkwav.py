"""mkwav.py SET N DIR: 16-bit WAVs and refs for the first N utterances of a packed corpus.
Reads the original FLAC when it is on disk, else decodes the packed mu-law bytes."""
import json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mulaw
s, n, d = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
d.mkdir(parents=True, exist_ok=True)
idx = json.load(open(f'../data/{s}.index.json'))[:n]
flac = Path(f'../data/LibriSpeech/{s}')
b = np.memmap(f'../data/{s}.bytes', dtype=np.uint8, mode='r')
src = 'flac' if flac.exists() else 'mu-law'
for u in idx:
    out = d / (u['id'] + '.wav')
    if out.exists():
        continue
    if flac.exists():
        spk, ch, _ = u['id'].split('-')
        x, sr = sf.read(flac / spk / ch / (u['id'] + '.flac'), dtype='float32')
    else:
        x, sr = mulaw.decode(np.asarray(b[u['offset']:u['offset'] + u['length']])), 16000
    sf.write(out, x, sr, subtype='PCM_16')
(d / 'refs.txt').write_text('\n'.join(u['text'].upper() for u in idx) + '\n')
(d / 'list.txt').write_text('\n'.join(u['id'] for u in idx) + '\n')
print(d, len(idx), 'utterances from', src)
