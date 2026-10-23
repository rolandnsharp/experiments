#!/usr/bin/env bash
# prof_harness.py at Diakosmos's size on faster GPUs, on Roland's own account (Roland, 2026-10-23: "we
# can even run on a more powerful pod"): one launcher per GPU kind, each with its own markers and pod
# name, so they run side by side and beside other sessions' launchers.
#   bash ~/field/launch-dk-prof.sh a100|4090|l40s|h100
set -uo pipefail
cd ~/field
case "${1:-}" in
  a100) GPUS="NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB"; CONFIGS="8x8c,16x4c,32x2c,8x8,16x4,8x8c+compile,16x4+compile" ;;
  4090) GPUS="4090"; CONFIGS="8x8c,16x4c,32x2c,8x8,4x16,8x8c+compile,16x4c+compile" ;;
  l40s) GPUS="NVIDIA L40S"; CONFIGS="8x8c,16x4c,32x2c,8x8,16x4,8x8c+compile,16x4+compile" ;;
  h100) GPUS="NVIDIA H100 80GB HBM3,NVIDIA H100 PCIe,NVIDIA H100 NVL"; CONFIGS="8x8c,16x4c,32x2c,8x8,16x4,8x8c+compile,16x4+compile" ;;
  *) echo "usage: launch-dk-prof.sh a100|4090|l40s|h100"; exit 1 ;;
esac
M=dk-prof-$1
mkdir -p $M && rm -f $M/LEGS-RESOLVED
export RP_HOME=$HOME/.runpod-roland POD_NAME=roland-dkprof BUSY_PREFIX=roland-dkprof-${1:0:4} MARKS=$M GPUS CLOUDS="COMMUNITY SECURE"
export LEGS="script:prof_harness.py" UPDATES=3000 LAYERS=14 HARMONICS=95 BATCH=8 SHARDS=1 POD_HOURS=2 RESUME=0
export EXTRA="--configs $CONFIGS --updates 3 --profile 8x8c" TAG=$1
nohup bash ~/field/pod-self.sh dkprof > ~/field/$M/pod-self-launch.log 2>&1 < /dev/null & disown
echo "$(date -u +%H:%M:%S) profiling launcher on Roland's account: $1 (diary ~/field/$M/pod-self.log)" | tee -a pause-big.log
