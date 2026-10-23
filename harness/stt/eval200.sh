#!/usr/bin/env bash
# The C port on the first 200 test-clean utterances (the same ones as decode_eval.py), float vs int8, greedy vs LM.
cd ~/zara/stt/port || exit 1
gcc -O3 -ffast-math -o zara_stt zara_stt.c -lm || exit 1
mkdir -p wav200 && ~/zara/.venv/bin/python - <<'PY'
import json, soundfile as sf
from pathlib import Path
idx = [u for u in json.load(open('../data/test-clean.index.json'))][:200]
refs = []
for u in idx:
    spk, ch, _ = u['id'].split('-')
    out = Path('wav200', u['id'] + '.wav')
    if not out.exists():
        x, sr = sf.read(f"../data/LibriSpeech/test-clean/{spk}/{ch}/{u['id']}.flac", dtype='float32')
        sf.write(out, x, sr, subtype='PCM_16')
    refs.append(u['text'].upper())
open('wav200/refs.txt', 'w').write('\n'.join(refs) + '\n')
open('wav200/list.txt', 'w').write('\n'.join(u['id'] for u in idx) + '\n')
PY
M=mel-mlp448-trained.bin; LM=../lm/librispeech-3gram.zlm
for mode in f32 int8; do
  q=""; [ $mode = int8 ] && q="--int8"
  for dec in greedy lm; do
    l=""; [ $dec = lm ] && l="--lm $LM --beam 32"
    cat wav200/list.txt | xargs -P 5 -I{} sh -c "./zara_stt $M wav200/{}.wav --chunk 16 $q $l > wav200/{}.$mode.$dec.txt"
    for id in $(cat wav200/list.txt); do cat wav200/$id.$mode.$dec.txt; done > hyp200-$mode-$dec.txt
    echo "$mode $dec: $(python3 score_lines.py wav200/refs.txt hyp200-$mode-$dec.txt)"
  done
done > eval200.txt 2>&1
echo done >> eval200.txt
