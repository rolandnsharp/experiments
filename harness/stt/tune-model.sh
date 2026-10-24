#!/usr/bin/env bash
# LM weights tuned on dev-clean (first 300), then all of test-clean scored once with them, int8 C port.
set -euo pipefail
cd ~/zara/stt/port
M=${M:-osc-mlp448-r4.bin}; T=${T:-r4}; LM=../lm/librispeech-3gram.zlm; PY=~/zara/.venv/bin/python
for d in dev300 test2620; do mkdir -p $d-$T; cp $d/list.txt $d/refs.txt $d-$T/; for f in $d/*.wav; do ln -sf ../$f $d-$T/; done; done
for d in dev300-$T test2620-$T; do
  xargs -P 5 -I{} sh -c "./zara_stt $M $d/{}.wav --chunk 16 --int8 --dump $d/{}.lp > $d/{}.greedy" < $d/list.txt
  $PY catlogp.py $d
  for id in $(cat $d/list.txt); do cat $d/$id.greedy; done > $d/h-greedy.txt
done
: > tune-$T.txt
for a in 0.3 0.5 0.7; do for b in 0.0 0.5 1.0 2.0; do
  ./zara_stt --decode-logp dev300-$T/all.logp --lm $LM --beam 32 --alpha $a --beta $b > dev300-$T/h-$a-$b.txt
  echo "dev alpha $a beta $b: $(python3 score_lines.py dev300-$T/refs.txt dev300-$T/h-$a-$b.txt)" >> tune-$T.txt
done; done
best=$(sort -t' ' -k7 -n tune-$T.txt | head -1)
a=$(echo "$best" | cut -d' ' -f3); b=$(echo "$best" | cut -d' ' -f5 | tr -d ':')
echo "chosen on dev: alpha $a beta $b ($best)" >> tune-$T.txt
echo "dev-clean 300, greedy: $(python3 score_lines.py dev300-$T/refs.txt dev300-$T/h-greedy.txt)" >> tune-$T.txt
echo "test-clean 2620, greedy, int8 C port: $(python3 score_lines.py test2620-$T/refs.txt test2620-$T/h-greedy.txt)" >> tune-$T.txt
for B in 32 64; do
  ./zara_stt --decode-logp test2620-$T/all.logp --lm $LM --beam $B --alpha $a --beta $b > test2620-$T/h-lm-b$B.txt
  echo "test-clean 2620, beam $B + 3-gram, alpha $a beta $b: $(python3 score_lines.py test2620-$T/refs.txt test2620-$T/h-lm-b$B.txt)" >> tune-$T.txt
done
echo done >> tune-$T.txt
