#!/usr/bin/env bash
# The full Diakosmos-size polar run on Roland's own RunPod account (Roland, 2026-10-23: "for that
# last big run lets use my pod account not ajaxs"): the polar arm of the 640-update check continued
# to 15,360 updates from its checkpoint, scored on the full held-out set against Diakosmos's 1.4194.
# Run by hand after the 640-update verdict:
#   bash ~/field/launch-dk-full.sh            # prints the plan and the cost, starts nothing
#   bash ~/field/launch-dk-full.sh go         # starts it
# The launcher runs as `pod-self.sh roland` (an argument, so chain-v2.sh's wait for its own launcher
# does not wait for this one), with its markers and diary in ~/field/roland/.
set -uo pipefail
cd ~/field
RUN=runs/polar-dense-14L95H-s0-u15360-b8-dk
[ -f $RUN/stopped-640.json ] && [ -f $RUN/resume.pt ] || { echo "no 640-update checkpoint in $RUN yet"; exit 1; }
RP_HOME=$HOME/.runpod-roland
K=$(grep -i apikey $RP_HOME/.runpod/config.toml | sed 's/.*= *//; s/["'"'"']//g' | tr -d ' ')
plan=$(K="$K" python3 - "$RUN" <<'PY'
import json, os, sys, urllib.request
run = sys.argv[1]
s = json.load(open(f'{run}/stopped-640.json'))
# the step time: the training clock at update 640 less the held-out checks' share (one 64-window check)
per = s['curve'][-1]['seconds'] / 640
left = 15360 - 640
hours = left * per / 3600 * 1.08 + 1.0           # 8% for the checks and pushes, an hour for the fetch and the final scoring
price = 0.22
cost = hours * price
q = 'query { myself { clientBalance currentSpendPerHr } }'
key = os.environ['K']
r = json.load(urllib.request.urlopen(urllib.request.Request(f'https://api.runpod.io/graphql?api_key={key}',
    data=json.dumps({'query': q}).encode(), headers={'Content-Type': 'application/json'}), timeout=60))['data']['myself']
other = r['currentSpendPerHr']
need = cost + other * hours + 3.0                 # this run, whatever else runs meanwhile, and $3 of margin
print(f"{per:.2f} {hours:.1f} {cost:.2f} {r['clientBalance']:.2f} {other:.3f} {need:.2f} {int(hours * 1.25) + 2}")
PY
) || { echo "could not size the run"; exit 1; }
read per hours cost balance other need watchdog <<< "$plan"
echo "step ${per} s; ${hours} h for the remaining 14,720 updates; about \$${cost} at \$0.22/h (community 3090)"
echo "account balance \$${balance}; other pods spending \$${other}/h; needed with margin \$${need}; watchdog ${watchdog} h"
python3 -c "import sys; sys.exit(0 if float('$balance') >= float('$need') else 1)" || { echo "NOT ENOUGH CREDIT: top up the account or stop the other pod first"; exit 1; }
[ "${1:-}" = go ] || { echo "plan only; run with 'go' to start"; exit 0; }
mkdir -p roland && rm -f roland/LEGS-RESOLVED
export RP_HOME BUSY_PREFIX=roland-field MARKS=roland
export LEGS="polar:dense:15360:0" UPDATES=15360 LAYERS=14 HARMONICS=95 BATCH=8 SHARDS=5 POD_HOURS=$watchdog RESUME=1
export EXTRA="--half --accumulate 8 --eval-every 640 --eval-windows 64 --final-windows 0" TAG=dk
nohup bash ~/field/pod-self.sh roland > ~/field/roland/pod-self-launch.log 2>&1 < /dev/null & disown
echo "$(date -u +%H:%M:%S) full Diakosmos-size polar run started on Roland's account (diary ~/field/roland/pod-self.log)" | tee -a pause-big.log
