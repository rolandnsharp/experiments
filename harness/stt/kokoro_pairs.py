"""Speak Thus Spake Zarathustra with Kokoro, one segment at a time, keeping the text.

    python kokoro_pairs.py TEXT OUT [--threads 6] [--test-fraction 0.05]

As ~/zara/tts.py (voice af_heart, speed 1.0, the same paragraphs) but every
piece Kokoro yields is kept as its own utterance with its graphemes and
phonemes, resampled 24 kHz -> 16 kHz (resample_poly 2/3, as the continuation
data) and mu-law encoded. Writes OUT/zara-train.bytes + .index.json and
OUT/zara-test.bytes + .index.json: the last TEST_FRACTION of the paragraphs
are the test set, which is the end of the book, where the continuation
model's held-out audio also comes from. Resumable: progress is kept in
OUT/progress.json after every paragraph.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mulaw  # noqa: E402

RATE = 16000


def paragraphs(path):
    text = open(path, encoding='utf-8').read()
    text = text[text.index('*** START'):]
    text = text[text.index('\n') + 1:]
    text = text[:text.index('*** END')]
    out = []
    for block in text.split('\n\n'):
        block = ' '.join(block.split())
        if block:
            out.append(block)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('text')
    ap.add_argument('out', type=Path)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--test-fraction', type=float, default=0.05)
    ap.add_argument('--limit', type=int, default=0, help='stop after this many new paragraphs')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    paras = paragraphs(args.text)
    first_test = int(round(len(paras) * (1 - args.test_fraction)))
    prog = args.out / 'progress.json'
    state = json.loads(prog.read_text()) if prog.exists() else {'next': 0, 'train': [], 'test': [], 'trainBytes': 0, 'testBytes': 0}
    print(f'{len(paras)} paragraphs, test from {first_test}, resuming at {state["next"]}', flush=True)
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    files = {s: open(args.out / f'zara-{s}.bytes', 'ab') for s in ('train', 'test')}
    done, t0 = 0, time.time()
    for n in range(state['next'], len(paras)):
        split = 'test' if n >= first_test else 'train'
        for i, (graphemes, phonemes, audio) in enumerate(pipeline(paras[n], voice='af_heart', speed=1.0)):
            if audio is None:
                continue
            x = resample_poly(audio.numpy().astype(np.float64), 2, 3)
            peak = float(np.abs(x).max()) or 1.0
            b = mulaw.encode(x * (0.9 / peak))
            files[split].write(b.tobytes())
            state[split].append({'id': f'zara-{n:05d}-{i:02d}', 'paragraph': n, 'offset': state[split + 'Bytes'],
                                 'length': int(b.size), 'text': graphemes.strip(), 'phonemes': phonemes.strip()})
            state[split + 'Bytes'] += int(b.size)
        files[split].flush()
        state['next'] = n + 1
        prog.write_text(json.dumps(state))
        done += 1
        if done % 20 == 0:
            h = (state['trainBytes'] + state['testBytes']) / RATE / 3600
            print(f'paragraph {n}: {h:.2f} h so far, {time.time() - t0:.0f} s this run', flush=True)
        if args.limit and done >= args.limit:
            break
    for s in ('train', 'test'):
        (args.out / f'zara-{s}.index.json').write_text(json.dumps(state[s]) + '\n')
        print(f'{s}: {len(state[s])} utterances, {state[s + "Bytes"] / RATE / 3600:.2f} h', flush=True)
    if state['next'] >= len(paras):
        (args.out / 'KOKORO-DONE').write_text('done\n')


if __name__ == '__main__':
    main()
