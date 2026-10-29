#!/usr/bin/env bash
# EXP-0048: this project's block + AdamW at round 3's shape and budget, one pod.
cd ~/zara/stt2
TAG=stt8 RESULTS=runs8 FRONTS=osc MLP=448 LAYERS=4 WIDTH=448 STATES=0 UPDATES=30000 EVAL_EVERY=1000 BATCH_SECONDS=240 \
  AUTOSTOP=72000 POD_DISK=100 POD_MEM=24 \
  CARDS="3090|NVIDIA GeForce RTX 3090|community
4090|NVIDIA GeForce RTX 4090|community
a40|NVIDIA A40|secure" nohup bash retry6.sh > retry-stt8.log 2>&1 < /dev/null & disown
echo launched
