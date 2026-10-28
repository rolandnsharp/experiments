#!/usr/bin/env bash
# Round 6 (EXP-0043): 10.4 M osc model (6 blocks, MLP 672), 150,000 updates from scratch, one pod.
cd ~/zara/stt
TAG=stt7 RESULTS=runs7 FRONTS=osc MLP=672 LAYERS=6 HARMONICS=32 UPDATES=150000 EVAL_EVERY=2500 BATCH_SECONDS=240 \
  AUTOSTOP=230000 POD_DISK=100 POD_MEM=32 \
  CARDS="4090|NVIDIA GeForce RTX 4090|community
a40|NVIDIA A40|secure
a6000|NVIDIA RTX A6000|secure" nohup bash retry6.sh > retry-stt7.log 2>&1 < /dev/null & disown
echo launched
