#!/usr/bin/env bash
# Round 2: long timescales. Rent ONE RTX 3070 pod, score yesterday's three models on 1 s
# windows, train the continuous-time variant (decays 0.1 ms .. 2 s) and a polar control on
# 1 s windows, copy everything back, stop the pod. Run on the lab:
#   nohup bash ~/zara/bytes/pod-run2.sh > ~/zara/bytes/pod-run2.log 2>&1 < /dev/null & disown
# Results land in ~/zara/poles2/<variant>/ and ~/zara/poles2/rescore.jsonl ; ~/zara/bytes/POD2-DONE marks the end.
set -uo pipefail
cd ~/zara/bytes
rm -f POD2-DONE
GPU_TAG=${GPU_TAG:-3070}
export POD_NAME=roland-poles-$GPU_TAG POD_GPU="${POD_GPU:-NVIDIA GeForce RTX $GPU_TAG}" POD_COST=0.5 POD_DISK=20 POD_ENVFILE=$HOME/.runpod/roland-poles.env
export PATH=$HOME/.local/bin:$PATH
UPDATES=${UPDATES:-3000}
WINDOW=${WINDOW:-16384}
BATCH=${BATCH:-4}
VARIANTS=${VARIANTS:-"ct polar"}
bash pod-up.sh || { echo "pod-up failed" ; echo fail > POD2-DONE; exit 1; }
source "$POD_ENVFILE"
SSHK=~/.runpod/ssh/runpodctl-ssh-key
OPTS="-i $SSHK -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o LogLevel=ERROR"
pod() { timeout "${T:-600}" ssh $OPTS -p "$POD_PORT" root@"$POD_IP" "$@"; }
stop_pod() { echo "== stopping pod $POD_ID"; runpodctl stop pod "$POD_ID"; }
trap 'stop_pod' EXIT

echo "== arming self-stop (7 h)"
scp -q $OPTS -P "$POD_PORT" ~/.pod-autostop.sh root@"$POD_IP":/root/pod-autostop.sh
pod 'nohup sh /root/pod-autostop.sh 25200 > /root/autostop.log 2>&1 < /dev/null & disown; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader'

echo "== uploading data, scripts and yesterday's params"
pod 'mkdir -p /root/zara/bytes /root/zara/poles2'
tar cf - mulaw16k zara_poles.py diakosmos_oracle.py mulaw.py | timeout 1800 ssh $OPTS -p "$POD_PORT" root@"$POD_IP" 'cd /root/zara/bytes && tar xf -' || { echo "upload failed"; echo fail > POD2-DONE; exit 1; }
( cd ~/zara && tar cf - poles/real/params.pt poles/polar/params.pt poles/cartesian/params.pt ) | timeout 600 ssh $OPTS -p "$POD_PORT" root@"$POD_IP" 'cd /root/zara && tar xf -' || { echo "params upload failed"; echo fail > POD2-DONE; exit 1; }
pod 'du -sh /root/zara/bytes/mulaw16k; ls /root/zara/bytes /root/zara/poles/*'
fetch() { timeout "${1:-120}" ssh $OPTS -p "$POD_PORT" root@"$POD_IP" 'cd /root/zara && tar cf - poles2 poles2-*.log 2>/dev/null' | tar xf - -C ~/zara/; }
pod 'cd /root/zara/bytes && python -c "import scipy" 2>/dev/null || pip install -q scipy; python -c "import torch, scipy; print(torch.__version__, torch.cuda.is_available(), scipy.__version__)"'

echo "== starting: rescore at window $WINDOW, then train $VARIANTS"
pod "cd /root/zara/bytes && rm -f RUN2-DONE && nohup bash -c 'for v in real polar cartesian; do python zara_poles.py score --variant \$v --params /root/zara/poles/\$v/params.pt --bytes mulaw16k --window $WINDOW --batch 2 --out /root/zara/poles2/rescore.jsonl >> /root/zara/poles2-rescore.log 2>&1; echo rescore-\$v exit=\$? >> RUN2-DONE; done; for v in $VARIANTS; do python zara_poles.py train --variant \$v --bytes mulaw16k --out /root/zara/poles2/\$v --layers 4 --harmonics 32 --batch $BATCH --window $WINDOW --updates $UPDATES --eval-every 500 --eval-windows 64 --checkpoint > /root/zara/poles2-\$v.log 2>&1; echo \$v exit=\$? >> RUN2-DONE; done; echo all >> RUN2-DONE' > /root/zara/run2.log 2>&1 < /dev/null & disown; echo started"

mkdir -p ~/zara/poles2
while true; do
  sleep 120
  done_lines=$(T=60 pod 'cat /root/zara/bytes/RUN2-DONE 2>/dev/null' 2>/dev/null)
  fetch 120 2>/dev/null
  for v in $VARIANTS; do [ -f ~/zara/poles2/$v/train.log ] && echo "-- $v: $(tail -1 ~/zara/poles2/$v/train.log)"; done
  echo "$done_lines" | grep -q '^all' && break
done
echo "== runs finished: $done_lines"
fetch 900
echo "$done_lines" > POD2-DONE
echo "== results in ~/zara/poles2"
