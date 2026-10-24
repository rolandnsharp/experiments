#!/usr/bin/env bash
# greedy int8 forward-pass time, one A53 core, 12 test-clean files (86.85 s audio)
cd ~/zara-stt/cand
for m in "$@"; do
  s=$(date +%s.%N)
  for w in ../bench/*.wav; do taskset -c 0 ../port/zara_stt $m $w --chunk 16 --int8 > /dev/null; done
  e=$(date +%s.%N)
  echo "$m $(echo "$e - $s" | bc) s, RTF $(echo "scale=3; ($e - $s)/86.85" | bc)"
done
