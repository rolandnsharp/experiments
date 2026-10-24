#!/usr/bin/env bash
# Round 5 (experiments/2026-10-24-stt-capacity) plus the seed-1 rerun, three pods in parallel on
# Roland s own RunPod account (~$21 in all). Check the balance first.
cd ~/zara/stt
BIG="a40|NVIDIA A40|secure
a6000|NVIDIA RTX A6000|secure"
TAG=stt6a RESULTS=runs6a FRONTS=osc MLP=672 LAYERS=6 HARMONICS=32 AUTOSTOP=72000 CARDS="$BIG" POD_DISK=100 POD_MEM=48 UPDATES=30000 BATCH_SECONDS=240 \
  nohup bash retry6.sh > retry-stt6a.log 2>&1 < /dev/null & disown
TAG=stt6b RESULTS=runs6b FRONTS=osc MLP=896 LAYERS=8 HARMONICS=32 AUTOSTOP=108000 CARDS="$BIG" POD_DISK=100 POD_MEM=48 UPDATES=30000 BATCH_SECONDS=240 \
  nohup bash retry6.sh > retry-stt6b.log 2>&1 < /dev/null & disown
TAG=stt5b RESULTS=runs5 FRONTS="osc mel" MLP=448 LAYERS=4 HARMONICS=32 SEED=1 AUTOSTOP=144000 POD_DISK=100 POD_MEM=24 UPDATES=30000 BATCH_SECONDS=240 \
  CARDS="3090|NVIDIA GeForce RTX 3090|community
a40|NVIDIA A40|secure" nohup bash retry6.sh > retry-stt5b.log 2>&1 < /dev/null & disown
echo launched; sleep 2; pgrep -fa "retry6.sh" | head
