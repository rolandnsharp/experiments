#!/usr/bin/env bash
# The polar 640-update leg moved to Roland's account (compiled, on a 4090; launch-dk-full.sh first), so the
# shared 3090's queued polar leg is not needed (Roland, 2026-10-23). When the shared pod's diary says the
# real leg is done (pushed), stop its launcher (pid in $1) before it can rent a replacement, stop and
# remove the pod, and write POD-DONE so chain-v4.sh goes on to the 6L/64H resume.
set -uo pipefail
cd ~/field
log() { echo "$(date -u +%H:%M:%S) $*" >> pause-big.log; }
until grep -q "leg real-dense-14L95H-s0-u15360-b8-dk done" pod-self.log 2>/dev/null; do sleep 30; done
kill "$1" 2>/dev/null
PID=$(cut -d" " -f2 POD-ID)
~/.local/bin/runpodctl stop pod "$PID" >> pause-big.log 2>&1
sleep 30
~/.local/bin/runpodctl remove pod "$PID" >> pause-big.log 2>&1
echo "real-dense-14L95H-s0-u15360-b8-dk exit=0; the polar leg moved to Roland's account (dkc)" > POD-DONE
log "shared pod $PID stopped after the real 640 leg: $(cut -c1-120 runs/real-dense-14L95H-s0-u15360-b8-dk/stopped-640.json 2>/dev/null)"
