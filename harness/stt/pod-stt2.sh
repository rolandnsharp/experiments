#!/usr/bin/env bash
# Recogniser round 2: LibriSpeech 960 h, mel first then osc, on ONE RTX 3090 (GPU_TAG=3080/3070
# also work). Waits while another roland-* pod is running (one pod at a time). Everything long
# runs detached on the pod (pod-run-all.sh); this driver only polls and fetches, so ssh drops
# do not matter. Run on the lab:
#   nohup bash ~/zara/stt/pod-stt2.sh > ~/zara/stt/pod-stt2.log 2>&1 < /dev/null & disown
# Results: ~/zara/stt/$RESULTS/<front>/ ; ~/zara/stt/POD-STT2-DONE marks the end.
# Rents on ROLAND'S OWN RunPod account: runpodctl config and ssh key live under RP_HOME
# (~/.runpod-roland), separate from Ajax's ~/.runpod. Never print or copy either key.
set -uo pipefail
cd ~/zara/stt
rm -f POD-STT2-DONE
RP_HOME=/home/roland/.runpod-roland
rp() { HOME=$RP_HOME runpodctl "$@"; }
GPU_TAG=${GPU_TAG:-3090}
export POD_NAME=${POD_NAME:-roland-stt2-$GPU_TAG} POD_GPU="${POD_GPU:-NVIDIA GeForce RTX $GPU_TAG}" POD_COST=0.6 POD_DISK=${POD_DISK:-100} POD_MEM=${POD_MEM:-48} POD_ENVFILE=$RP_HOME/.runpod/roland-stt2.env
export PATH=$HOME/.local/bin:$PATH
SUBSETS=${SUBSETS:-"test-clean test-other train-clean-100 train-clean-360 train-other-500"}
TRAIN=${TRAIN:-train-clean-100,train-clean-360,train-other-500}
FRONTS=${FRONTS:-"osc mel"}
UPDATES=${UPDATES:-30000}
EVAL_EVERY=${EVAL_EVERY:-1000}
BATCH_SECONDS=${BATCH_SECONDS:-240}
AUTOSTOP=${AUTOSTOP:-50400}

while rp get pod 2>/dev/null | grep -v "^ID" | grep -q RUNNING; do
  echo "$(date -u +%H:%M) another pod is running on this account, waiting 10 min"; sleep 600
done
HOME=$RP_HOME bash ~/zara/bytes/pod-up.sh || { echo "pod-up failed" ; echo fail > POD-STT2-DONE; exit 1; }
source "$POD_ENVFILE"
SSHK=$RP_HOME/.runpod/ssh/runpodctl-ssh-key
OPTS="-i $SSHK -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o LogLevel=ERROR"
pod() { timeout "${T:-120}" ssh $OPTS -p "$POD_PORT" root@"$POD_IP" "$@"; }
stop_pod() { echo "== stopping pod $POD_ID"; rp stop pod "$POD_ID"; }
trap 'stop_pod' EXIT

echo "== arming self-stop ($AUTOSTOP s)"
scp -q $OPTS -P "$POD_PORT" ~/.pod-autostop.sh root@"$POD_IP":/root/pod-autostop.sh
pod "nohup sh /root/pod-autostop.sh $AUTOSTOP > /root/autostop.log 2>&1 < /dev/null & disown; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; nproc; df -h /root | tail -1"

echo "== uploading scripts and the Kokoro test pairs"
pod 'mkdir -p /root/zara/stt/data /root/zara/stt/runs'
EXTRA_TESTS=test-other
EXTRA=""; [ -f data/zara-test.bytes ] && [ -s data/zara-test.bytes ] && { EXTRA="data/zara-test.bytes data/zara-test.index.json"; EXTRA_TESTS=test-other,zara-test; }
for i in 1 2 3; do
  tar cf - zara_stt.py zara_poles.py diakosmos_oracle.py mulaw.py pack_libri.py pod-run-all.sh $EXTRA | timeout 3600 ssh $OPTS -p "$POD_PORT" root@"$POD_IP" 'cd /root/zara/stt && tar xf -' && break
  echo "upload attempt $i failed"; sleep 30
done
pod 'cd /root/zara/stt && ls -la zara_stt.py pod-run-all.sh data/ && pip install -q scipy soundfile 2>&1 | grep -v -i "warning\|notice"'
CUDA=$(pod 'python -c "import torch; print(torch.__version__, torch.cuda.is_available())" 2>/dev/null' | tail -1)
echo "== torch: $CUDA"
if ! echo "$CUDA" | grep -q True; then
  echo "cuda unavailable on this host: removing pod $POD_ID and giving up on it"
  rp remove pod "$POD_ID"; trap - EXIT; echo fail > POD-STT2-DONE; exit 1
fi

echo "== starting on the pod: subsets [$SUBSETS] train $TRAIN fronts [$FRONTS] updates $UPDATES batch $BATCH_SECONDS s extra tests $EXTRA_TESTS"
pod "cd /root/zara/stt && rm -f RUN-DONE DATA-DONE && SUBSETS='$SUBSETS' TRAIN='$TRAIN' FRONTS='$FRONTS' UPDATES=$UPDATES EVAL_EVERY=$EVAL_EVERY BATCH_SECONDS=$BATCH_SECONDS EXTRA_TESTS=$EXTRA_TESTS MLP=${MLP:-0} nohup bash pod-run-all.sh > run-all.log 2>&1 < /dev/null & disown; echo started"

fetch() { timeout "${1:-300}" ssh $OPTS -p "$POD_PORT" root@"$POD_IP" 'cd /root/zara/stt && tar cf - runs runs-*.log run-all.log 2>/dev/null' | tar xf - -C ~/zara/stt/$RESULTS/; }
mkdir -p ~/zara/stt/runs2
RESULTS=${RESULTS:-runs2}
while true; do
  sleep 300
  done_lines=$(pod 'cat /root/zara/stt/RUN-DONE 2>/dev/null' 2>/dev/null)
  fetch 300 2>/dev/null
  [ -f ~/zara/stt/$RESULTS/run-all.log ] && echo "$(date -u +%H:%M) -- data: $(grep -v '^$' ~/zara/stt/$RESULTS/run-all.log | tail -1 | cut -c1-120)"
  for f in $FRONTS; do [ -f ~/zara/stt/$RESULTS/runs/$f/train.log ] && echo "$(date -u +%H:%M) -- $f: $(tail -1 ~/zara/stt/$RESULTS/runs/$f/train.log | cut -c1-200)"; done
  echo "$done_lines" | grep -q '^all' && break
done
echo "== runs finished: $done_lines"
fetch 1800
echo "$done_lines" > POD-STT2-DONE
echo "== results in ~/zara/stt/$RESULTS/runs"
