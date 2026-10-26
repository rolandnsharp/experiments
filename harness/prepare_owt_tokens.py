#!/usr/bin/env python3
"""Byte-level BPE tokens of the OpenWebText byte shards, for field_poles.py --tokens
(experiment 2026-10-27-bytes-vs-subword, EXP-0033).

    prepare_owt_tokens.py --bytes owt --vocab 2048 --shards 2 --out owt-bpe2k [--tokenizer-bytes 100000000]

The tokenizer is BPE over bytes: each byte is one symbol (GPT-2's printable mapping of the 256 bytes),
so every byte sequence encodes and decodes exactly, whatever its UTF-8. It is trained on the first
--tokenizer-bytes of training shard 0 only, after splitting the text before words, numbers, runs of
other symbols and runs of whitespace (a word keeps its leading space, as GPT-2's split does). Id 0 is
the start token (U+2402, which no text maps to), covering no bytes; the other vocab - 1 ids are the 256 bytes and the merges.

Written to OUT: train.u16, the first --shards training shards as one token stream in manifest order
(each shard cut into pieces of about 1 MB after a space or newline and encoded piece by piece);
heldout.npz, the held-out set as field_poles.py's byte windows score it: window k predicts stream bytes
k*2048+1 .. k*2048+2048 given byte k*2048 (the stream is heldout.bytes from contextBytes - 1, as
TextStream builds it), so here window k is those 2048 bytes (fewer in the last window: only the scored
bytes, not the zero padding), tokenized alone; with the bytes each window covers, every token's bytes
and the three sample prompts tokenized; tokenizer.json; tokens.json, the manifest. Every held-out window
is checked to decode to its bytes, and the training stream to cover each shard's bytes exactly.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from tokenizers import Regex, Tokenizer, models, pre_tokenizers, trainers

T = 2048
PIECE = 1 << 20
# the start token: a symbol the byte mapping never produces, so no text can contain it (a special token's
# name is matched inside the text, and a literal '<s>' in OpenWebText would become a token of no bytes)
START = '\u2402'


def byte_symbols():
    """GPT-2's bytes_to_unicode: the printable bytes as themselves, the rest from U+0100 on."""
    keep = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    chars, n = {}, 0
    for b in range(256):
        if b in keep:
            chars[b] = chr(b)
        else:
            chars[b] = chr(256 + n)
            n += 1
    return chars


B2S = byte_symbols()
S2B = {s: b for b, s in B2S.items()}
TRANSLATE = {b: s for b, s in B2S.items()}
WS = ''.join(B2S[b] for b in b' \n\t\r')
# a word, a number or a run of other symbols, each with an optional leading space; or a run of whitespace
PATTERN = f'{B2S[32]}?[A-Za-z]+|{B2S[32]}?[0-9]+|{B2S[32]}?[^A-Za-z0-9{WS}]+|[{WS}]+'


def mapped(b):
    return bytes(b).decode('latin-1').translate(TRANSLATE)


def pieces(data):
    """Cut a byte array into pieces of about PIECE bytes, each ending after a space or newline when one
    is near; together exactly the array."""
    out, a = [], 0
    while a < data.size:
        b = min(a + PIECE, data.size)
        if b < data.size:
            tail = data[a + PIECE // 2:b]
            cut = np.flatnonzero((tail == 32) | (tail == 10))
            if cut.size:
                b = a + PIECE // 2 + int(cut[-1]) + 1
        out.append(data[a:b])
        a = b
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bytes', type=Path, required=True)
    ap.add_argument('--vocab', type=int, required=True)
    ap.add_argument('--shards', type=int, default=2)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tokenizer-bytes', type=int, default=100_000_000)
    args = ap.parse_args()
    t0 = time.time()
    m = json.loads((args.bytes / 'manifest.json').read_text())
    shards = m['shards'][:args.shards]
    args.out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(PATTERN), behavior='isolated')
    trainer = trainers.BpeTrainer(vocab_size=args.vocab, special_tokens=[START], initial_alphabet=list(B2S.values()),
                                  min_frequency=2, show_progress=False)
    first = np.fromfile(args.bytes / shards[0]['train']['file'], dtype=np.uint8, count=args.tokenizer_bytes)
    tok.train_from_iterator((mapped(p) for p in pieces(first)), trainer)
    if tok.get_vocab_size() != args.vocab or tok.token_to_id(START) != 0:
        raise SystemExit(f'vocabulary {tok.get_vocab_size()} (asked {args.vocab}), the start token at {tok.token_to_id(START)}')
    tok.save(str(args.out / 'tokenizer.json'))
    symbols = [tok.id_to_token(i) for i in range(args.vocab)]
    vocab_bytes = [b'' if s == START else bytes(S2B[ch] for ch in s) for s in symbols]
    lengths = np.array([len(b) for b in vocab_bytes], dtype=np.int64)
    print(f'tokenizer trained on {first.size} bytes of {shards[0]["train"]["file"]}: {args.vocab} ids ({time.time() - t0:.0f} s)', flush=True)

    def encode(chunks):
        return [np.array(e.ids, dtype=np.uint16) for e in tok.encode_batch([mapped(c) for c in chunks], add_special_tokens=False)]

    train_tokens, train_bytes = 0, 0
    with open(args.out / 'train.u16', 'wb') as f:
        for s in shards:
            data = np.fromfile(args.bytes / s['train']['file'], dtype=np.uint8)
            parts = pieces(data)
            covered = 0
            for i in range(0, len(parts), 64):
                for ids in encode(parts[i:i + 64]):
                    covered += int(lengths[ids].sum())
                    train_tokens += ids.size
                    f.write(ids.tobytes())
            if covered != data.size:
                raise SystemExit(f'{s["train"]["file"]}: tokens cover {covered} bytes of {data.size}')
            train_bytes += data.size
            print(f'{s["train"]["file"]}: {data.size} bytes, {train_tokens} tokens so far ({time.time() - t0:.0f} s)', flush=True)

    h = m['heldout']
    text = np.fromfile(args.bytes / h['file'], dtype=np.uint8)
    count = -(-h['scoredBytes'] // T)
    stream = np.zeros(count * T + 1, dtype=np.uint8)
    stream[:text.size - h['contextBytes'] + 1] = text[h['contextBytes'] - 1:]
    windows = [stream[k * T + 1:k * T + 1 + min(T, h['scoredBytes'] - k * T)] for k in range(count)]
    ids = encode(windows)
    for k, (w, t) in enumerate(zip(windows, ids)):
        if b''.join(vocab_bytes[i] for i in t) != bytes(w):
            raise SystemExit(f'held-out window {k} does not decode to its bytes')
    nbytes = np.array([w.size for w in windows], dtype=np.int64)
    if nbytes.sum() != h['scoredBytes']:
        raise SystemExit(f'held-out windows cover {nbytes.sum()} bytes, the manifest scores {h["scoredBytes"]}')
    span = text.size - 2 * 256
    prompts = [text[(i + 1) * span // 4:(i + 1) * span // 4 + 256] for i in range(3)]   # TextStream.prompts
    held_tokens = np.concatenate(ids)
    np.savez(args.out / 'heldout.npz', tokens=held_tokens, offsets=np.concatenate([[0], np.cumsum([t.size for t in ids])]).astype(np.int64),
             nbytes=nbytes, vocab_bytes=np.frombuffer(b''.join(vocab_bytes), dtype=np.uint8),
             vocab_offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
             **{f'prompt{i}': encode([p])[0].astype(np.int64) for i, p in enumerate(prompts)})

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'schema': 'vidya-owt-bpe/1', 'vocab': args.vocab, 'bos': 0, 'shards': len(shards),
                'train': 'train.u16', 'trainTokens': train_tokens, 'trainBytes': train_bytes,
                'trainBytesPerToken': train_bytes / train_tokens,
                'heldout': 'heldout.npz', 'heldoutWindows': count, 'heldoutTokens': int(held_tokens.size),
                'heldoutBytes': int(nbytes.sum()), 'heldoutBytesPerToken': float(nbytes.sum() / held_tokens.size),
                'tokenizer': 'tokenizer.json', 'tokenizerSha256': sha(args.out / 'tokenizer.json'),
                'tokenizerTrainedOn': f'the first {first.size} bytes of {shards[0]["train"]["file"]}',
                'pattern': PATTERN, 'heldoutSha256': sha(args.out / 'heldout.npz'),
                'sourceManifestSha256': sha(args.bytes / 'manifest.json'), 'seconds': time.time() - t0}
    (args.out / 'tokens.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if 'PerToken' in k or 'Tokens' in k}), flush=True)


if __name__ == '__main__':
    os.environ.setdefault('RAYON_NUM_THREADS', '6')
    main()
