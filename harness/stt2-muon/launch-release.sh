#!/usr/bin/env bash
# EXP-0049: release candidate, 10.36 M (6 blocks, MLP 672), own block + Muon, 150,000 updates, one pod.
cd ~/zara/stt2-muon
TAG=stt11 RESULTS=runs11 LR=3e-3 WARMUP=200 OPTIMIZER=muon FRONTS=osc MLP=672 LAYERS=6 WIDTH=448 STATES=0 UPDATES=150000 EVAL_EVERY=2500 BATCH_SECONDS=240 \
  AUTOSTOP=198000 POD_DISK=100 POD_MEM=24 \
  CARDS="3090|NVIDIA GeForce RTX 3090|community
4090|NVIDIA GeForce RTX 4090|community
a40|NVIDIA A40|secure" nohup bash retry6.sh > retry-stt11.log 2>&1 < /dev/null & disown
echo launched
