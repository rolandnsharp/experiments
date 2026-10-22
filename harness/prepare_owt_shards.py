#!/usr/bin/env python3
"""More OpenWebText training shards for an existing prepare_owt_bytes.py output.

    prepare_owt_shards.py --output DIR --first 2 --last 9

Fetches parquet files first..last of Skylion007/openwebtext (the pinned
revision), writes each as train-NNNNN.bytes by prepare_owt_bytes.py's rule
(every document that is NOT held out, as UTF-8 text then <|endoftext|>), and
appends the shard to manifest.json's `shards` (and `sources`) as each one
completes, so a reader of the first N shards sees the same stream whether N
shards or more were prepared. The held-out set is untouched: its documents
are the ones hashed out of every file, including these.
"""
import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from prepare_owt_bytes import FILES, SEPARATOR, fetch, held_out, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--first', type=int, required=True)
    p.add_argument('--last', type=int, required=True)
    a = p.parse_args()
    manifest = a.output / 'manifest.json'
    tmp = a.output / 'tmp'
    tmp.mkdir(exist_ok=True)
    for i in range(a.first, a.last + 1):
        m = json.loads(manifest.read_text())
        have = [s['source'] for s in m['shards']]
        name = f'train-{i:05d}-of-{FILES:05d}.parquet'
        if name in have:
            print(f'{name}: already in the manifest', flush=True)
            continue
        if have != [f'train-{j:05d}-of-{FILES:05d}.parquet' for j in range(len(have))] or len(have) != i:
            raise SystemExit(f'the manifest holds {have}; shard {i} must follow them in order')
        path, _ = fetch(i, tmp)
        digest = sha256_file(path)
        texts = pq.read_table(path, columns=['text']).column('text').to_pylist()
        partial = a.output / f'train-{i:05d}.bytes.partial'
        train_docs = 0
        file_val = 0
        with open(partial, 'wb') as out:
            for text in texts:
                b = text.encode('utf-8')
                if held_out(b):
                    file_val += 1
                else:
                    out.write(b)
                    out.write(SEPARATOR)
                    train_docs += 1
        final = a.output / f'train-{i:05d}.bytes'
        os.replace(partial, final)
        path.unlink()
        m['shards'].append({'source': name, 'train': {'file': final.name, 'bytes': final.stat().st_size,
                            'documents': train_docs, 'sha256': sha256_file(final)}})
        sources = {s['file']: s for s in m['sources']}
        sources[name] = {'file': name, 'sha256': digest, 'documents': len(texts), 'heldOutDocuments': file_val}
        m['sources'] = [sources[k] for k in sorted(sources)]
        m['trainBytes'] = sum(s['train']['bytes'] for s in m['shards'])
        m['trainDocuments'] = sum(s['train']['documents'] for s in m['shards'])
        manifest.with_suffix('.json.partial').write_text(json.dumps(m, indent=1) + '\n')
        os.replace(manifest.with_suffix('.json.partial'), manifest)
        print(f'{name}: {len(texts)} documents, {file_val} held out, {train_docs} to train-{i:05d}.bytes ({final.stat().st_size} bytes)', flush=True)
    tmp.rmdir()
    print('done', flush=True)


if __name__ == '__main__':
    main()
