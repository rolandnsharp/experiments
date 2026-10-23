#!/usr/bin/env bash
# After the real-decay 30K control (seed 0): Diakosmos's recipe at his size for 640 updates (real decay,
# then the polar pole; scored on the full held-out set against his 1.9806 at 640), then the 6L/64H
# resume, then the control's seed 1. Replaces chain-after-control.sh (2026-10-23).
set -uo pipefail
cd ~/field
log() { echo "$(date -u +%H:%M:%S) $*" >> pause-big.log; }
wait_pod() {  # until the current launcher has written POD-DONE and exited
  sleep 300
  while [ ! -f POD-DONE ]; do sleep 120; done
  while pgrep -f "bash .*pod-self.sh$" > /dev/null; do sleep 30; done
}
launch() { nohup bash ~/field/pod-self.sh > ~/field/pod-self-launch.log 2>&1 < /dev/null & disown; }
wait_pod
log "control finished: $(tr "\n" " " < POD-DONE)"
mkdir -p control-s0 && cp pod-self.log control-s0/pod-self.log && cp pod-self-launch.log control-s0/launch.log
rm -f LEGS-RESOLVED
export LEGS="real:dense:15360:0 polar:dense:15360:0" UPDATES=15360 LAYERS=14 HARMONICS=95 BATCH=8 SHARDS=1 POD_HOURS=6 RESUME=0
export EXTRA="--half --accumulate 8 --stop-at 640 --eval-every 640 --eval-windows 64 --final-windows 0" TAG=dk
launch
log "Diakosmos-recipe 640-update launcher started (real, polar; 14L/95H, 64 windows an update, half)"
wait_pod
log "Diakosmos-recipe 640 finished: $(tr "\n" " " < POD-DONE)"
mkdir -p dk640 && cp pod-self.log dk640/pod-self.log && cp pod-self-launch.log dk640/launch.log
export EXTRA="" TAG=""
printf "polar:dense:69154:0 16\n" > LEGS-RESOLVED
export LEGS="polar:dense:69154:0" UPDATES=3000 LAYERS=6 HARMONICS=64 BATCH=16 SHARDS=10 POD_HOURS=20 RESUME=1
launch
log "6L/64H resume launcher started"
wait_pod
log "6L/64H finished: $(tr "\n" " " < POD-DONE)"
mkdir -p big-run-part2 && cp pod-self.log big-run-part2/pod-self.log && cp pod-self-launch.log big-run-part2/launch.log
rm -f LEGS-RESOLVED
export LEGS="real:dense:30000:1" UPDATES=3000 LAYERS=4 HARMONICS=32 BATCH=16 SHARDS=2 POD_HOURS=10 RESUME=0
launch
log "control seed 1 launcher started"
