#!/usr/bin/env python3
"""OpenWebText as bytes, with Diakosmos's held-out set reproduced from the source.

    prepare_owt_bytes.py --output DIR [--keep-train-files 2] [--windows 4096]

Streams the 80 parquet files of Skylion007/openwebtext (revision pinned below)
in name order, one at a time, deleted after use. A document is held out when
little-endian sha256(text)[:4] % 2000 == 0 (diakosmos-prepare-bytes.py's rule,
the GPT-2 baseline's val split); every other document of the first
--keep-train-files files goes to the training stream, each document's UTF-8
text then the 13 bytes of <|endoftext|>, as train-NNNNN.bytes.

The held-out documents of every file, in order, are GPT-2 BPE tokenized
(tiktoken gpt2; each document's ids then 50256) and the first
1024 x --windows + 1 ids are decoded to heldout.bytes; offset k (k = 0 .. W)
is where id 1024 k + 1 begins, so contextBytes = offset 0 and the scored bytes
are [offset 0, offset W). manifest.json carries Diakosmos's keys, so
diakosmos-train-windows.py and the pole harness read it unchanged. Ajax's
run scored 18,538,212 bytes over 4,096 windows; a reproduction must give the
same count.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken

REVISION = '79d93d786212f7344586290adb811d4ae6a1762c'
FILES = 80
SEPARATOR = b'<|endoftext|>'
END_OF_TEXT = 50256
CONTEXT = 1024
HELD_OUT_MODULUS = 2000
# lsalpha.com/training/diakosmos: the GPT-2 baseline's held-out input, 4,096 windows
EXPECTED_INPUT_SHA256 = 'd068497406362f161eac476b733bca5e0fd525f7e1f8f896bb0fe546eb5ae5ba'


def held_out(text_bytes):
    return int.from_bytes(hashlib.sha256(text_bytes).digest()[:4], 'little') % HELD_OUT_MODULUS == 0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def fetch(i, tmp):
    name = f'train-{i:05d}-of-{FILES:05d}.parquet'
    url = f'https://huggingface.co/datasets/Skylion007/openwebtext/resolve/{REVISION}/plain_text/{name}'
    path = tmp / name
    for attempt in range(5):
        r = subprocess.run(['curl', '-sSL', '--retry', '3', '--max-time', '900', '-o', str(path), url])
        if r.returncode == 0 and path.stat().st_size > 1000:
            return path, name
    raise SystemExit(f'could not download {name}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--keep-train-files', type=int, default=2)
    p.add_argument('--windows', type=int, default=4096)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'manifest.json').unlink(missing_ok=True)
    tmp = a.output / 'tmp'
    tmp.mkdir(exist_ok=True)
    enc = tiktoken.get_encoding('gpt2')
    need = CONTEXT * a.windows + 1
    val_ids = []
    val_docs = 0
    shards = []
    sources = []
    for i in range(FILES):
        path, name = fetch(i, tmp)
        digest = sha256_file(path)
        table = pq.read_table(path, columns=['text'])
        texts = table.column('text').to_pylist()
        train_out = open(a.output / f'train-{i:05d}.bytes.partial', 'wb') if i < a.keep_train_files else None
        train_docs = 0
        file_val = 0
        for text in texts:
            b = text.encode('utf-8')
            if held_out(b):
                file_val += 1
                if len(val_ids) < need:
                    val_ids.extend(enc.encode_ordinary(text))
                    val_ids.append(END_OF_TEXT)
            elif train_out is not None:
                train_out.write(b)
                train_out.write(SEPARATOR)
                train_docs += 1
        val_docs += file_val
        if train_out is not None:
            train_out.close()
            final = a.output / f'train-{i:05d}.bytes'
            os.replace(a.output / f'train-{i:05d}.bytes.partial', final)
            shards.append({'source': name, 'train': {'file': final.name, 'bytes': final.stat().st_size,
                           'documents': train_docs, 'sha256': sha256_file(final)}})
        sources.append({'file': name, 'sha256': digest, 'documents': len(texts), 'heldOutDocuments': file_val})
        path.unlink()
        print(f'{name}: {len(texts)} documents, {file_val} held out, val ids so far {len(val_ids)}', flush=True)
        if len(val_ids) >= need and i >= a.keep_train_files - 1 and i == FILES - 1:
            break
    if len(val_ids) < need:
        raise SystemExit(f'only {len(val_ids)} held-out ids, {need} needed')
    ids = val_ids[:need]
    lengths = np.array([len(enc.decode_single_token_bytes(t)) for t in ids], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)])
    text = b''.join(enc.decode_single_token_bytes(t) for t in ids)
    if len(text) != int(starts[-1]):
        raise SystemExit('the held-out text decodes to a length its ids do not give')
    offsets = starts[1 + CONTEXT * np.arange(a.windows + 1)].astype('<u8')
    (a.output / 'heldout.bytes').write_bytes(text)
    offsets.tofile(a.output / 'heldout-windows.u64')
    # the GPT-2 baseline's held-out input: window k = ids [1024 k, 1024 k + 1025), little-endian u32
    arr = np.array(ids, dtype=np.int64)
    windows = np.stack([arr[CONTEXT * k:CONTEXT * k + CONTEXT + 1] for k in range(a.windows)]).astype('<u4')
    windows.tofile(a.output / 'heldout-input.u32')
    input_sha = sha256_file(a.output / 'heldout-input.u32')
    print('held-out input sha256', input_sha, 'MATCH' if input_sha == EXPECTED_INPUT_SHA256 else 'MISMATCH (expected ' + EXPECTED_INPUT_SHA256 + ')', flush=True)
    heldout = {'file': 'heldout.bytes', 'bytes': len(text), 'sha256': hashlib.sha256(text).hexdigest(),
               'windows': a.windows, 'tokens': len(ids), 'contextBytes': int(offsets[0]),
               'scoredBytes': int(offsets[-1] - offsets[0]), 'offsets': 'heldout-windows.u64',
               'offsetsSha256': sha256_file(a.output / 'heldout-windows.u64'),
               'idsSha256': hashlib.sha256(np.array(ids, dtype='<u2').tobytes()).hexdigest(),
               'input': 'heldout-input.u32', 'inputSha256': input_sha, 'inputMatchesBaseline': input_sha == EXPECTED_INPUT_SHA256}
    record = {
        'schema': 'diakosmos-openwebtext-bytes/1', 'dataset': 'Skylion007/openwebtext', 'revision': REVISION,
        'from': 'the parquet files directly (prepare_owt_bytes.py)',
        'separator': SEPARATOR.decode(), 'separatorRule': "each document's UTF-8 text, then the separator",
        'heldOutRule': 'little-endian sha256(text)[:4] % 2000 == 0',
        'trainBytes': sum(s['train']['bytes'] for s in shards), 'trainDocuments': sum(s['train']['documents'] for s in shards),
        'valDocuments': val_docs, 'heldout': heldout, 'shards': shards, 'sources': sources, 'complete': True,
    }
    (a.output / 'manifest.json').write_text(json.dumps(record, indent=1) + '\n')
    tmp.rmdir()
    print(json.dumps({k: v for k, v in record.items() if k not in ('shards', 'sources')}, indent=1))


if __name__ == '__main__':
    main()
