#!/usr/bin/env bash
# Keep trying to rent a pod for one recogniser job (pod-stt6.sh), every 10 minutes for up to 8 h.
# The job comes from the environment (TAG, RESULTS, FRONTS, MLP, LAYERS, WIDTH, STATES, LR, SEED, ...);
# CARDS lists "tag|runpod gpu type|community or secure" lines, tried in order.
#   TAG=stt6a ... nohup bash ~/zara/stt2-muon/retry6.sh > ~/zara/stt2-muon/retry-stt6a.log 2>&1 < /dev/null & disown
cd ~/zara/stt2-muon
: "${TAG:?set TAG}" "${CARDS:?set CARDS}"
DONE=POD-${TAG^^}-DONE; LOG=pod-$TAG.log
for i in $(seq 1 48); do
  while IFS="|" read -r gtag gpu cloud; do
    rm -f "$DONE"
    sec=0; [ "$cloud" = secure ] && sec=1
    POD_NAME=roland-$TAG-$gtag-$i GPU_TAG=$gtag POD_GPU="$gpu" POD_SECURE=$sec bash ~/zara/stt2-muon/pod-stt6.sh > "$LOG" 2>&1
    if grep -q "^fail" "$DONE" 2>/dev/null && grep -q "no pod id\|no longer any instances\|cuda unavailable" "$LOG"; then
      echo "$(date -u +%H:%M) $gtag ($cloud): $(grep -o "no longer any instances\|no pod id\|cuda unavailable on this host" "$LOG" | head -1) (attempt $i)"
      cp "$LOG" "pod-$TAG-attempt$i-$gtag.log"; continue
    fi
    echo "$(date -u +%H:%M) finished on $gtag: $(cat "$DONE" 2>/dev/null | tr "\n" " ")"; exit 0
  done <<< "$CARDS"
  sleep 600
done
echo "gave up"
