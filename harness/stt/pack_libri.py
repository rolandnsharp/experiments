"""Pack a LibriSpeech subset into one mu-law byte file plus an index.

    python pack_libri.py LIBRISPEECH/SUBSET OUT/NAME [JOBS]

Writes OUT/NAME.bytes (every utterance's 16 kHz samples, mu-law, one byte per
sample, concatenated) and OUT/NAME.index.json: a list of
{"id", "offset", "length", "text"} in corpus order, text upper-case letters,
space and apostrophe as LibriSpeech gives it. One global gain is not applied:
each utterance is scaled to peak 0.9 before encoding, since recordings differ.
"""
import json
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mulaw  # noqa: E402

RATE = 16000


def pack_chapter(trans):
    """One chapter's utterances -> (list of (uid, text, mu-law bytes))."""
    out = []
    for line in trans.read_text().splitlines():
        uid, text = line.split(' ', 1)
        x, sr = sf.read(trans.parent / (uid + '.flac'), dtype='float32')
        assert sr == RATE and x.ndim == 1, (uid, sr, x.shape)
        peak = float(np.abs(x).max()) or 1.0
        out.append((uid, text.strip(), mulaw.encode(x * (0.9 / peak)).tobytes()))
    return out


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, os.cpu_count() // 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    index, offset = [], 0
    chapters = sorted(src.rglob('*.trans.txt'))
    with open(str(out) + '.bytes', 'wb') as f, multiprocessing.Pool(jobs) as pool:
        for chapter in pool.imap(pack_chapter, chapters, chunksize=4):   # imap keeps corpus order
            for uid, text, b in chapter:
                f.write(b)
                index.append({'id': uid, 'offset': offset, 'length': len(b), 'text': text})
                offset += len(b)
    (Path(str(out) + '.index.json')).write_text(json.dumps(index) + '\n')
    hours = offset / RATE / 3600
    print(f'{out}: {len(index)} utterances, {hours:.2f} h, {offset} bytes', flush=True)


if __name__ == '__main__':
    main()
