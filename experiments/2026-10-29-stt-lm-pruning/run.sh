#!/usr/bin/env bash
# EXP-0042: 3-gram pruned 3e-7 vs 1e-7 on round 3 logits (dev300, test2620), same protocol as tune-model.sh
set -euo pipefail
cd ~/zara/stt/port
PY=~/zara/.venv/bin/python
if [ ! -f ../lm/librispeech-3gram-1e-7.zlm ]; then
  curl -sSL https://openslr.elda.org/resources/11/3-gram.pruned.1e-7.arpa.gz | zcat > lmexp/1e-7.arpa
  /usr/bin/time -v $PY export_lm.py lmexp/1e-7.arpa ../lm/librispeech-3gram-1e-7.zlm 2> lmexp/export-1e-7.time
  rm lmexp/1e-7.arpa
fi
for arm in 3e-7 1e-7; do
  LM=../lm/librispeech-3gram.zlm; [ $arm = 1e-7 ] && LM=../lm/librispeech-3gram-1e-7.zlm
  out=lmexp/tune-$arm.txt; : > $out
  for a in 0.3 0.5 0.7; do for b in 0.0 0.5 1.0 2.0; do
    ./zara_stt --decode-logp dev300/all.logp --lm $LM --beam 32 --alpha $a --beta $b > lmexp/dev-$arm-$a-$b.txt
    echo "dev alpha $a beta $b: $(python3 score_lines.py dev300/refs.txt lmexp/dev-$arm-$a-$b.txt)" >> $out
  done; done
  best=$(sort -t" " -k7 -n $out | head -1)
  a=$(echo "$best" | cut -d" " -f3); b=$(echo "$best" | cut -d" " -f5 | tr -d :)
  echo "chosen on dev: alpha $a beta $b ($best)" >> $out
  for B in 32 64; do
    /usr/bin/time -f "%e s %M KB" ./zara_stt --decode-logp test2620/all.logp --lm $LM --beam $B --alpha $a --beta $b > lmexp/test-$arm-b$B.txt 2> lmexp/test-$arm-b$B.time
    echo "test-clean 2620, beam $B, alpha $a beta $b: $(python3 score_lines.py test2620/refs.txt lmexp/test-$arm-b$B.txt) [$(tail -1 lmexp/test-$arm-b$B.time)]" >> $out
  done
  echo "lm $(stat -c %s $LM) bytes" >> $out
done
echo done > lmexp/DONE
