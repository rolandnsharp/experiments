#!/usr/bin/env bash
# CPU sanity check, not an experiment: AdamW vs Muon on the own block, round-3 shape, same data, rate and budget
A="train --front osc --data /home/roland/zara/stt/data --train dev-clean --test test-clean --mlp 448 --layers 4 --poles 64 --batch-seconds 30 --updates 600 --eval-every 200 --eval-utterances 60 --test-limit 60 --checkpoint --lr 3e-3 --warmup 60"
cd ~/zara/stt2-muon
for o in adamw muon; do OMP_NUM_THREADS=4 nohup ~/zara/.venv/bin/python zara_stt.py $A --optimizer $o --out /tmp/cmp-$o > /tmp/cmp-$o.log 2>&1 & done
wait
