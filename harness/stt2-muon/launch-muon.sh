#!/usr/bin/env bash
# EXP-0051: EXP-0050 (3e-3, warmup 200) with Muon on the blocks matrices, one pod.
cd ~/zara/stt2-muon
TAG=stt10 RESULTS=runs10 LR=3e-3 WARMUP=200 OPTIMIZER=muon FRONTS=osc MLP=448 LAYERS=4 WIDTH=448 STATES=0 UPDATES=30000 EVAL_EVERY=1000 BATCH_SECONDS=240 \
  AUTOSTOP=72000 POD_DISK=100 POD_MEM=24 \
  CARDS="3090|NVIDIA GeForce RTX 3090|community
4090|NVIDIA GeForce RTX 4090|community
a40|NVIDIA A40|secure" nohup bash retry6.sh > retry-stt10.log 2>&1 < /dev/null & disown
echo launched
