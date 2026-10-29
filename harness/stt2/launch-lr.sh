#!/usr/bin/env bash
# EXP-0050: EXP-0048 at round 3's learning rate and warmup (3e-3, 200 updates), one pod.
cd ~/zara/stt2
TAG=stt9 RESULTS=runs9 LR=3e-3 WARMUP=200 FRONTS=osc MLP=448 LAYERS=4 WIDTH=448 STATES=0 UPDATES=30000 EVAL_EVERY=1000 BATCH_SECONDS=240 \
  AUTOSTOP=72000 POD_DISK=100 POD_MEM=24 \
  CARDS="3090|NVIDIA GeForce RTX 3090|community
4090|NVIDIA GeForce RTX 4090|community
a40|NVIDIA A40|secure" nohup bash retry6.sh > retry-stt9.log 2>&1 < /dev/null & disown
echo launched
