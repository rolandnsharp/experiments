#!/usr/bin/env bash
# CPU sanity check, not an experiment: new block (stt2) vs old (stt), round-3 shape, same data and budget
A="train --front osc --data /home/roland/zara/stt/data --train dev-clean --test test-clean --mlp 448 --layers 4 --poles 64 --batch-seconds 30 --updates 600 --eval-every 200 --eval-utterances 60 --test-limit 60 --checkpoint"
cd ~/zara/stt2 && OMP_NUM_THREADS=4 nohup ~/zara/.venv/bin/python zara_stt.py $A --out /tmp/cmp-new > /tmp/cmp-new.log 2>&1 &
cd ~/zara/stt && OMP_NUM_THREADS=4 nohup ~/zara/.venv/bin/python zara_stt.py $A --harmonics 32 --out /tmp/cmp-old > /tmp/cmp-old.log 2>&1 &
wait
