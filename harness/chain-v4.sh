#!/usr/bin/env bash
# chain-v3.sh without the profiling job (it runs on Roland's account on faster GPUs, 2026-10-23), and
# waiting on its own launcher's process rather than on every pod-self.sh (another session runs
# launchers of its own): the 640-update check's launcher (pid in the first argument), then the
# 6L/64H resume, then the 30K control's seed 1, all on the shared account.
set -uo pipefail
cd ~/field
log() { echo "$(date -u +%H:%M:%S) $*" >> pause-big.log; }
wait_pid() {  # until launcher $1 has exited and POD-DONE is written
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
  while [ ! -f POD-DONE ]; do sleep 60; done
}
launch() { nohup bash ~/field/pod-self.sh > ~/field/pod-self-launch.log 2>&1 < /dev/null & LAUNCHER=$!; }
wait_pid "$1"
log "Diakosmos-recipe 640 finished: $(tr "\n" " " < POD-DONE)"
for d in runs/*-14L95H-s0-u15360-b8-dk; do [ -f $d/stopped-640.json ] && log "$d: $(cut -c1-120 $d/stopped-640.json)"; done
mkdir -p dk640 && cp pod-self.log dk640/pod-self.log && cp pod-self-launch.log dk640/launch.log
export EXTRA="" TAG="" MARKS=. GPUS=3070,3080,3090 CLOUDS=COMMUNITY
printf "polar:dense:69154:0 16\n" > LEGS-RESOLVED
export LEGS="polar:dense:69154:0" UPDATES=3000 LAYERS=6 HARMONICS=64 BATCH=16 SHARDS=10 POD_HOURS=20 RESUME=1
launch
log "6L/64H resume launcher started (pid $LAUNCHER)"
wait_pid "$LAUNCHER"
log "6L/64H finished: $(tr "\n" " " < POD-DONE)"
mkdir -p big-run-part2 && cp pod-self.log big-run-part2/pod-self.log && cp pod-self-launch.log big-run-part2/launch.log
rm -f LEGS-RESOLVED
export LEGS="real:dense:30000:1" UPDATES=3000 LAYERS=4 HARMONICS=32 BATCH=16 SHARDS=2 POD_HOURS=10 RESUME=0
launch
log "control seed 1 launcher started (pid $LAUNCHER)"
