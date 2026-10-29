#!/usr/bin/env bash
# Stage a Hugging Face release on the lab: release-build.sh PARAMS.pt NUMBERS.json OUTDIR
# Exports the C model, copies the public code (release-kit/public/: only this project's code), writes
# dictate.sh with the dev-tuned LM weights, fills the card, self-tests the C program from the
# staged files only, and writes SHA256SUMS.
set -euo pipefail
PT=$1 NUMS=$2 OUT=$3
KIT=$(cd "$(dirname "$0")" && pwd); PY=~/zara/.venv/bin/python
rm -rf "$OUT"; mkdir -p "$OUT"
$PY "$KIT/public/export_c.py" export "$PT" "$OUT/model.bin"
cp "$PT" "$OUT/params.pt"
cp ~/zara/stt/lm/librispeech-3gram.zlm "$OUT/"
cp "$KIT"/public/{zara_stt.c,zara_stt.py,zara_block.py,mulaw.py,pack_libri.py,export_c.py,export_lm.py,score_lines.py} "$OUT/"
! { grep -il "diakosmos\|oracle\|harmonic.gpt" "$OUT"/*.py "$OUT"/zara_stt.c "$OUT"/README.md; grep -l "\bAlpha\b" "$OUT"/*.py "$OUT"/zara_stt.c "$OUT"/README.md; } || { echo "a name from outside this project found in the release"; exit 1; }
A=$($PY -c "import json;print(json.load(open('$NUMS'))['alpha'])"); B=$($PY -c "import json;print(json.load(open('$NUMS'))['beta'])")
cat > "$OUT/dictate.sh" <<EOT
#!/bin/sh
# Live dictation on the uConsole: headset mic (ES8388, stereo 44.1 kHz, voice = L - R) -> the
# oscillator recogniser (int8) -> 3-gram beam search on each pause. Ctrl-C stops.
cd "\$(dirname "\$0")" || exit 1
[ -x zara_stt ] || cc -O3 -ffast-math -o zara_stt zara_stt.c -lm || exit 1
arecord -q -D hw:1 -f S16_LE -r 44100 -c 2 -t raw | ./zara_stt model.bin --stream --mic --int8 --chunk 16 --lm librispeech-3gram.zlm --beam 32 --alpha $A --beta $B
EOT
chmod +x "$OUT/dictate.sh"
$PY "$KIT/fill_card.py" "$KIT/README.template.md" "$NUMS" "$OUT/README.md"
# self-test: build from the staged C file alone and transcribe a held-out clip
T=$(mktemp -d); cp "$OUT/zara_stt.c" "$T/"; (cd "$T" && cc -O3 -ffast-math -o zara_stt zara_stt.c -lm 2> cc.log) || { cat "$T/cc.log"; exit 1; }
W=~/zara/stt/port/dev300/$(head -1 ~/zara/stt/port/dev300/list.txt).wav
echo "self-test greedy: $("$T/zara_stt" "$OUT/model.bin" "$W" --int8 --chunk 16)"
echo "self-test LM:     $("$T/zara_stt" "$OUT/model.bin" "$W" --int8 --chunk 16 --lm "$OUT/librispeech-3gram.zlm" --alpha $A --beta $B)"
echo "reference:        $(sed -n 1p ~/zara/stt/port/dev300/refs.txt)"
[ -s "$T/cc.log" ] && { echo "compiler warnings:"; cat "$T/cc.log"; }
rm -rf "$T"
(cd "$OUT" && sha256sum $(ls | grep -v SHA256SUMS) > SHA256SUMS)
du -sh "$OUT"; ls -la "$OUT"
