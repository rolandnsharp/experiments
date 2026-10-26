#!/usr/bin/env bash
# Runs ON THE POD as its container command: no inbound SSH needed. The pod
# fetches the text harness and the OpenWebText bytes from the lab, runs the
# legs on the GPU, pushes every result back to the lab after each leg (and
# every 30 minutes while a leg runs), writes the lab's POD-DONE marker, and
# stops itself. A watchdog stops the pod after POD_HOURS hours whatever
# happens. Environment (from the deployment):
#   LAB_KEY_B64  the lab's pod-upload private key, base64
#   LEGS         "variant:mixing[:updates[:seed[:flags[:name]]]] ..." (default real:dense polar:dense polar:grouped);
#                flags: more train arguments for that leg only, comma-separated, a value after =
#                (--select,--harmonics=20,--mlp=600), after EXTRA; name: appended to its run directory
#                a leg "script:NAME.py" runs that lab script instead: python NAME.py --bytes owt
#                --out runs/NAME[-TAG] EXTRA, its outputs pushed like a leg's
#                updates "autoH" trains for H hours: a 30-update benchmark sets the count
#   UPDATES      updates a leg when the leg does not say (default 3000)
#   LAYERS HARMONICS BATCH SHARDS   the model and the data (default 4 32 16 0 = every shard)
#   RESUME       1: a leg whose runs/NAME/resume.pt is on the lab continues from it
#   a leg that fails with a checkpoint behind it is retried from it twice, then the host is handed back
#   POD_HOURS    watchdog (default 6)
#   EXTRA        more field_poles.py train arguments for every leg (default none), e.g. "--half --accumulate 8"
#   TAG          appended to every leg's run directory name (default none)
#   MARKS        the lab directory under ~/field for this launcher's markers and diary (POD-DONE,
#                POD-RETRY, LEGS-RESOLVED, pod-self.log; default ., ~/field itself), so launchers on
#                two accounts can run at once
set -uo pipefail
LAB=roland@158.69.195.38
LP=2222
LEGS=${LEGS:-"real:dense polar:dense polar:grouped"}
UPDATES=${UPDATES:-3000}
LAYERS=${LAYERS:-4}; HARMONICS=${HARMONICS:-32}; BATCH=${BATCH:-16}; SHARDS=${SHARDS:-0}; RESUME=${RESUME:-0}
POD_HOURS=${POD_HOURS:-6}
EXTRA=${EXTRA:-}; TAG=${TAG:-}; MARKS=${MARKS:-.}
mkdir -p /root/.ssh && chmod 700 /root/.ssh
echo "$LAB_KEY_B64" | base64 -d > /root/.ssh/lab && chmod 600 /root/.ssh/lab
SSH="ssh -i /root/.ssh/lab -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -p $LP"
lab() { $SSH $LAB "$@"; }
say() { echo "$(date -u +%H:%M:%S) $*"; lab "echo \"$(date -u +%H:%M:%S) $*\" >> ~/field/$MARKS/pod-self.log" 2>/dev/null; }
fail() { say "$*"; lab "echo fail > ~/field/$MARKS/POD-DONE"; runpodctl stop pod "$RUNPOD_POD_ID"; exit 1; }
# the watchdog: stop this pod after POD_HOURS hours
( sleep $(( POD_HOURS * 3600 )); runpodctl stop pod "$RUNPOD_POD_ID" ) > /root/watchdog.log 2>&1 &
say "pod $RUNPOD_POD_ID up: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
# a host whose container cannot reach CUDA is useless: say so, ask the launcher for another, stop
CUDA=$(python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.__version__)" 2>&1 | tail -1)
say "torch cuda: $CUDA"
case "$CUDA" in True*) ;; *) say "no CUDA in this container; asking for another host"; lab "echo $RUNPOD_POD_ID > ~/field/$MARKS/POD-RETRY"; runpodctl stop pod "$RUNPOD_POD_ID"; exit 1 ;; esac
say "fetching"
mkdir -p /root/field/runs && cd /root/field
# scripts, then the bytes the legs need as a tar stream
SCRIPTS=$(for l in $LEGS; do case "$l" in script:*) echo "${l#script:}" ;; esac; done)
for f in field_poles.py zara_poles.py diakosmos_oracle.py mulaw.py compute_cost.py $SCRIPTS; do
  lab "cat ~/field/$f" > /root/field/$f || fail "fetch $f failed"
done
FILES=$(lab "bash ~/field/owt-files.sh $SHARDS") || fail "listing the shards failed"
lab "cd ~/field && tar cf - $FILES" | tar xf - -C /root/field || fail "fetch owt failed"
# the token directories legs name with --tokens=DIR (prepare_owt_tokens.py's output on the lab)
for d in $(grep -o -- '--tokens=[^ ,:]*' <<< "$LEGS $EXTRA" | cut -d= -f2 | sort -u); do
  lab "cd ~/field && tar cf - $d" | tar xf - -C /root/field || fail "fetch $d failed"
  say "fetched $d ($(du -sh /root/field/$d | cut -f1))"
done
say "fetched $(du -sh /root/field/owt | cut -f1) of bytes; $(ls /root/field/owt | wc -l) files; model ${LAYERS}L/${HARMONICS}H batch $BATCH shards $SHARDS resume $RESUME"
python -c "import scipy" 2>/dev/null || pip install -q scipy
push() {  # copy the runs (results, logs, checkpoints) to the lab
  tar cf - -C /root/field runs | lab "cd ~/field && tar xf -" 2>/dev/null
}
# every 30 min, push whatever the run has so far (the harness saves the checkpoint at every held-out check)
( while true; do sleep 1800; push; say "pushed interim results ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1) used): $(for d in /root/field/runs/*/; do [ -f $d/train.log ] && echo "$(basename $d) $(grep held-out $d/train.log | tail -1 | cut -c1-60)"; done | tr "\n" ";")"; done ) > /root/pusher.log 2>&1 &
train() {  # train NAME VARIANT MIXING SEED UPDATES BATCH [extra args]
  local name=$1 v=$2 m=$3 sd=$4 u=$5 b=$6; shift 6
  python field_poles.py train --variant $v --mixing $m --seed $sd --bytes owt --out /root/field/runs/$name \
    --layers $LAYERS --harmonics $HARMONICS --batch $b --updates $u --eval-every 500 --checkpoint --shards $SHARDS $EXTRA "$@" \
    > /root/field/runs/$name.log 2>&1
}
# a leg is variant:mixing[:updates[:seed[:flags[:name]]]]; its run directory names the model, seed, updates, batch
# and the leg's name
for leg in $LEGS; do
  case "$leg" in
    script:*)
      s=${leg#script:}; name=${s%.py}${TAG:+-$TAG}
      say "script $s starting"
      python $s --bytes owt --out /root/field/runs/$name $EXTRA > /root/field/runs/$name.log 2>&1
      rc=$?
      echo "$name exit=$rc" >> /root/field/runs/RUN-DONE
      push
      say "script $s done (exit $rc): $(tail -1 /root/field/runs/$name.log | cut -c1-160)"
      continue ;;
  esac
  IFS=: read -r v m u sd lf ln <<< "$leg"; u=${u:-$UPDATES}; sd=${sd:-0}; b=$BATCH
  lflags=${lf//,/ }; lh=$(sed -n 's/.*--harmonics=\([0-9]*\).*/\1/p' <<< "$lf"); lh=${lh:-$HARMONICS}
  case "$u" in
    auto*)
      hours=${u#auto}
      # a 30-update benchmark at this batch (half of it when the memory does not hold it) sets the count
      for try in 1 2 3; do
        train bench-$v-$m-b$b $v $m $sd 30 $b --eval-every 1000 --final-windows 8
        if grep -q "out of memory" /root/field/runs/bench-$v-$m-b$b.log; then say "batch $b does not fit; halving"; b=$((b / 2)); continue; fi
        break
      done
      step=$(python - <<PY
import re
t = {}
for line in open("/root/field/runs/bench-$v-$m-b$b/train.log"):
    g = re.match(r"step\s+(\d+)\s+loss.*?\s(\d+) s$", line.rstrip())
    if g: t[int(g.group(1))] = int(g.group(2))
print((t[29] - t[0]) / 29 if 29 in t and 0 in t else 0)
PY
)
      [ "$step" = "0" ] && fail "the benchmark gave no timing: $(tail -3 /root/field/runs/bench-$v-$m-b$b.log | tr '\n' ' ')"
      u=$(python -c "print(int($hours * 3600 * 0.92 / $step))")
      say "benchmark: $step s an update at batch $b ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1) used); $hours h gives $u updates, $(python -c "print(round($u * $b * 2048 / 1e9, 2))") GB of bytes"
      ;;
  esac
  name=$v-$m-${LAYERS}L${lh}H-s$sd-u$u-b$b${ln:+-$ln}${TAG:+-$TAG}
  # the resolved leg, so a launcher that rents another host after a drop can continue it
  lab "echo \"$v:$m:$u:$sd${lf:+:$lf${ln:+:$ln}} $b\" >> ~/field/$MARKS/LEGS-RESOLVED"
  resume=""
  if [ "$RESUME" = "1" ] && lab "test -f ~/field/runs/$name/resume.pt"; then
    mkdir -p /root/field/runs/$name
    lab "cd ~/field && tar cf - runs/$name" | tar xf - -C /root/field && resume="--resume"
    say "leg $name continues from the lab's copy ($(cat /root/field/runs/$name/checkpoint.json 2>/dev/null | cut -c1-40))"
  fi
  say "leg $name starting"
  train $name $v $m $sd $u $b $lflags $resume
  rc=$?
  # a leg that dies with a checkpoint behind it (a CUDA fault on a community host, most often) continues
  # from the checkpoint, twice; a host that keeps failing is handed back to the launcher with RESUME
  tries=0
  while [ $rc -ne 0 ] && [ -f /root/field/runs/$name/resume.pt ] && [ $tries -lt 2 ]; do
    tries=$((tries + 1))
    say "leg $name failed (exit $rc): $(grep -m1 -E "Error|error" /root/field/runs/$name.log | cut -c1-120); retry $tries from its checkpoint ($(cut -c1-40 /root/field/runs/$name/checkpoint.json))"
    push
    mv /root/field/runs/$name.log /root/field/runs/$name.try$tries.log
    train $name $v $m $sd $u $b $lflags --resume
    rc=$?
  done
  if [ $rc -ne 0 ] && [ -f /root/field/runs/$name/resume.pt ]; then
    push
    say "leg $name failed three times on this host; asking the launcher for another host to continue from the checkpoint"
    lab "echo $RUNPOD_POD_ID > ~/field/$MARKS/POD-RETRY"
    runpodctl stop pod "$RUNPOD_POD_ID"
    exit 1
  fi
  echo "$name exit=$rc" >> /root/field/runs/RUN-DONE
  python compute_cost.py /root/field/runs/$name >> /root/field/runs/$name.log 2>&1
  push
  say "leg $name done: $(grep "final held-out" /root/field/runs/$name/train.log 2>/dev/null | tail -1)"
done
echo all >> /root/field/runs/RUN-DONE
push
lab "cat > ~/field/$MARKS/POD-DONE" < /root/field/runs/RUN-DONE
say "all legs done; stopping pod"
runpodctl stop pod "$RUNPOD_POD_ID"
