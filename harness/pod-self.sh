#!/usr/bin/env bash
# Rent ONE community RTX 3070, or a 3080 or 3090 when no 3070 is in stock (Ajax, sshmail #3282; Roland 2026-10-20), through the GraphQL
# API and let it drive itself: pod-self-driver.sh runs as the container
# command (it pulls the data from the lab, trains, pushes results back and
# stops the pod). Retries every 5 minutes for up to 8 hours. The pod writes
# the lab marker ~/field/POD-DONE at the end and keeps a diary in
# ~/field/pod-self.log. A pod that disappears mid-run (a community host going
# away) is replaced, up to four times, by another that continues the run from
# the lab's latest copy of its checkpoint (the driver's RESUME). Environment:
# LEGS UPDATES LAYERS HARMONICS BATCH SHARDS POD_HOURS EXTRA TAG as the driver reads them.
# RP_HOME: the RunPod account, as the home directory holding its .runpod/config.toml (default
# $HOME, the shared lab account; Roland's own is ~/.runpod-roland). BUSY_PREFIX: the pod names
# that count as this project's one running pod (default roland-, every roland pod). MARKS: the
# directory under ~/field for this launcher's markers and diary (default ., ~/field itself).
# Run on the lab:
#   nohup bash ~/field/pod-self.sh > ~/field/pod-self-launch.log 2>&1 < /dev/null & disown
set -uo pipefail
cd ~/field
export MARKS=${MARKS:-.}
mkdir -p "$MARKS" && cd "$MARKS"
rm -f POD-DONE POD-ID POD-RETRY pod-self.log
export LEGS=${LEGS:-"real:dense polar:dense polar:grouped"} UPDATES=${UPDATES:-3000} POD_HOURS=${POD_HOURS:-6}
export LAYERS=${LAYERS:-4} HARMONICS=${HARMONICS:-32} BATCH=${BATCH:-16} SHARDS=${SHARDS:-0} RESUME=${RESUME:-0}
export EXTRA=${EXTRA:-} TAG=${TAG:-}
export RP_HOME=${RP_HOME:-$HOME} BUSY_PREFIX=${BUSY_PREFIX:-roland-}
export RUNPOD_KEY=$(grep -i apikey "$RP_HOME/.runpod/config.toml" | sed 's/.*= *//; s/["'"'"']//g' | tr -d ' ')
export LAB_KEY_B64=$(base64 -w0 ~/.ssh/pod_upload)
export DRIVER_B64=$(base64 -w0 ~/field/pod-self-driver.sh)
gql() {  # one GraphQL query with a JSON variables object; prints the response
python3 - "$1" "${2:-{\}}" <<'PY'
import json, os, sys, urllib.request
key = os.environ['RUNPOD_KEY']
req = urllib.request.Request(f'https://api.runpod.io/graphql?api_key={key}', data=json.dumps({'query': sys.argv[1], 'variables': json.loads(sys.argv[2])}).encode(), headers={'Content-Type': 'application/json'})
try:
    print(json.dumps(json.load(urllib.request.urlopen(req, timeout=60))))
except Exception as e:
    print(json.dumps({'errors': [{'message': str(e)}]}))
PY
}
deploy() {  # deploy GPU; prints the pod id, or the error
python3 - "$1" <<'PY'
import json, os, sys, urllib.request
gpu = sys.argv[1]
key = os.environ['RUNPOD_KEY']
def q(query, variables=None):
    req = urllib.request.Request(f'https://api.runpod.io/graphql?api_key={key}', data=json.dumps({'query': query, 'variables': variables or {}}).encode(), headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=60))
pods = q('query { myself { pods { name desiredStatus } } }')['data']['myself']['pods']
busy = [p['name'] for p in pods if p['name'].startswith(os.environ['BUSY_PREFIX']) and p['desiredStatus'] == 'RUNNING']
if busy:
    print('busy ' + ' '.join(busy)); sys.exit(0)
env = [{'key': k, 'value': os.environ[k]} for k in ('LAB_KEY_B64', 'DRIVER_B64', 'LEGS', 'UPDATES', 'POD_HOURS', 'LAYERS', 'HARMONICS', 'BATCH', 'SHARDS', 'RESUME', 'EXTRA', 'TAG', 'MARKS')]
inp = dict(cloudType='COMMUNITY', gpuCount=1, volumeInGb=0, containerDiskInGb=30, minVcpuCount=1, minMemoryInGb=8,
           gpuTypeId=f'NVIDIA GeForce RTX {gpu}', name=f'roland-field-{gpu}',
           imageName='runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04',
           dockerArgs="bash -c 'echo $DRIVER_B64 | base64 -d > /root/driver.sh; bash /root/driver.sh > /root/driver.log 2>&1'",
           env=env, volumeMountPath='/workspace')
# the community cloud only (Ajax's rule)
for cloud in ('COMMUNITY',):
    inp['cloudType'] = cloud
    r = q('mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) { id machineId costPerHr machine { gpuDisplayName } } }', {'input': inp})
    d = (r.get('data') or {}).get('podFindAndDeployOnDemand')
    if d:
        break
if d:
    # a host that already failed CUDA is not tried again: remove the pod at once
    bad = set(open(os.path.expanduser('~/field/bad-machines')).read().split()) if os.path.exists(os.path.expanduser('~/field/bad-machines')) else set()
    if d.get('machineId') in bad:
        q('mutation { podTerminate(input:{podId:"%s"}) }' % d['id'])
        print('badhost ' + d['id'] + ' ' + str(d.get('machineId'))); sys.exit(0)
    print('pod ' + d['id'] + ' ' + d['machine']['gpuDisplayName'] + ' ' + str(d.get('machineId')) + ' $' + str(d.get('costPerHr')) + '/h ' + cloud)
else:
    print('error ' + '; '.join(e.get('message', '') for e in r.get('errors', [])))
PY
}
status() {  # the pod's desiredStatus, "gone" when RunPod no longer lists it, "unknown" when the API fails
  gql 'query($id: String!) { pod(input: {podId: $id}) { desiredStatus } }' "{\"id\": \"$1\"}" | python3 -c "
import json, sys
r = json.load(sys.stdin)
p = (r.get('data') or {}).get('pod') if 'errors' not in r else None
print('unknown' if 'errors' in r else (p or {}).get('desiredStatus', 'gone'))"
}
remove() { HOME=$RP_HOME ~/.local/bin/runpodctl remove pod "$1" 2>&1 | tail -1; }
resolved_legs() {  # the legs the driver resolved (LEGS-RESOLVED) that have no result yet; sets BATCH to theirs
  local v m u sd b name out=""
  while read -r spec b; do
    IFS=: read -r v m u sd <<< "$spec"; name=$v-$m-${LAYERS}L${HARMONICS}H-s$sd-u$u-b$b${TAG:+-$TAG}
    [ -f ~/field/runs/$name/result.json ] && continue
    stop=$(sed -n 's/.*--stop-at \([0-9]*\).*/\1/p' <<< "$EXTRA")
    [ -n "$stop" ] && [ -f ~/field/runs/$name/stopped-$stop.json ] && continue
    case " $out " in *" $spec "*) ;; *) out="$out $spec"; BATCH=$b ;; esac
  done < LEGS-RESOLVED
  echo $out
}
deadline=$(( $(date +%s) + 8 * 3600 ))
drops=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  for gpu in 3070 3080 3090; do
    R=$(deploy $gpu)
    case "$R" in
      pod\ *)
        echo "== $(date -u +%H:%M) rented $R; it drives itself (diary: ~/field/$MARKS/pod-self.log)"; echo "$R" > POD-ID
        PID=$(echo "$R" | cut -d" " -f2); MACHINE=$(echo "$R" | cut -d" " -f5)
        # wait for the pod's verdict: POD-DONE (finished), POD-RETRY (a host without CUDA: remove it, rent another),
        # or the pod vanishing mid-run (the host went away: rent another and continue from the lab's checkpoint)
        rm -f POD-RETRY
        waited=0; dropped=0
        while [ ! -f POD-DONE ] && [ ! -f POD-RETRY ] && [ $waited -lt $(( POD_HOURS * 3600 + 600 )) ]; do
          sleep 60; waited=$((waited + 60))
          if [ $(( waited % 300 )) -eq 0 ] && [ $waited -gt 600 ]; then
            st=$(status "$PID")
            case "$st" in RUNNING|unknown) ;; *) sleep 30; [ -f POD-DONE ] || [ -f POD-RETRY ] || { dropped=1; break; } ;; esac
          fi
        done
        if [ -f POD-RETRY ]; then
          echo "== $(date -u +%H:%M) pod $PID gave up on machine $MACHINE (no CUDA, or a leg that kept failing); removing it, remembering the machine, renting another"
          echo "$MACHINE" >> ~/field/bad-machines
          sleep 20; remove "$PID"; rm -f POD-RETRY POD-ID
          # legs the driver had resolved continue from their checkpoints
          if [ -f LEGS-RESOLVED ]; then export RESUME=1; export LEGS="$(resolved_legs)"; export BATCH; echo "== continuing legs:$LEGS at batch $BATCH from their checkpoints"; fi
          [ -f LEGS-RESOLVED ] && [ -z "${LEGS// /}" ] && { echo "every resolved leg has its result" | tee POD-DONE; exit 0; }
          sleep 120; continue 2
        fi
        if [ $dropped = 1 ]; then
          drops=$((drops + 1))
          echo "== $(date -u +%H:%M) pod $PID went away ($st) on machine $MACHINE without finishing (drop $drops)"
          echo "$(date -u +%H:%M:%S) launcher: pod $PID went away ($st); renting another to continue" >> pod-self.log
          [ "$st" = "gone" ] || remove "$PID"; rm -f POD-ID
          [ $drops -ge 4 ] && { echo "four drops; giving up" | tee POD-DONE; exit 1; }
          export RESUME=1
          [ -f LEGS-RESOLVED ] && { export LEGS="$(resolved_legs)"; export BATCH; echo "== continuing legs:$LEGS at batch $BATCH"; }
          [ -z "${LEGS// /}" ] && { echo "every resolved leg has its result" | tee POD-DONE; exit 0; }
          sleep 120; continue 2
        fi
        [ -f POD-DONE ] && { echo "== $(date -u +%H:%M) pod finished: $(cat POD-DONE | tr "\n" " ")"; sleep 45; echo "== removing the stopped pod: $(remove "$PID")"; exit 0; }
        echo "== $(date -u +%H:%M) no verdict from pod $PID within the watchdog; stopping it"; HOME=$RP_HOME ~/.local/bin/runpodctl stop pod "$PID" 2>&1 | tail -1; echo fail > POD-DONE; exit 1 ;;
      busy\ *) echo "another roland pod is running ($R); not starting"; echo fail > POD-DONE; exit 1 ;;
      badhost\ *) echo "== $(date -u +%H:%M) RTX $gpu: landed on a known bad machine ($R); removed" ;;
      *) echo "== $(date -u +%H:%M) RTX $gpu: $R" ;;
    esac
    sleep 10
  done
  sleep 300
done
echo "no pod in 8 hours" | tee POD-DONE
