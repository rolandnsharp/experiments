#!/usr/bin/env bash
# Keep trying to rent a pod for recogniser round 2, cheapest adequate card first, every 10 minutes
# for up to 8 hours. Cards of 24 GB or more take 240 s batches, smaller ones 120 s.
#   nohup bash ~/zara/stt/retry2.sh > ~/zara/stt/retry2.log 2>&1 < /dev/null & disown
cd ~/zara/stt
# tag | runpod gpu type | cloud (community/secure) | batch seconds | autostop seconds
CARDS='4090|NVIDIA GeForce RTX 4090|community|240|43200
3090|NVIDIA GeForce RTX 3090|community|240|50400
a40|NVIDIA A40|secure|240|50400
l4|NVIDIA L4|secure|240|57600
3080|NVIDIA GeForce RTX 3080|community|120|57600
3070|NVIDIA GeForce RTX 3070|community|120|57600'
for i in $(seq 1 48); do
  while IFS='|' read -r tag gpu cloud batch autostop; do
    rm -f POD-STT4-DONE
    sec=0; [ "$cloud" = secure ] && sec=1
    POD_NAME=roland-stt4-$tag-$i MLP=448 RESULTS=runs4 FRONTS=osc UPDATES=60000 AUGMENT=1 WORKERS=6 INIT_FILE=$HOME/zara/stt/runs3/runs/osc/params.pt GPU_TAG=$tag POD_GPU="$gpu" POD_SECURE=$sec BATCH_SECONDS=$batch AUTOSTOP=129600 bash ~/zara/stt/pod-stt4.sh > ~/zara/stt/pod-stt4.log 2>&1
    if grep -q "^fail" POD-STT4-DONE 2>/dev/null && grep -q "no pod id\|no longer any instances\|cuda unavailable" pod-stt4.log; then
      echo "$(date -u +%H:%M) $tag ($cloud): $(grep -o "no longer any instances\|no pod id\|cuda unavailable on this host" pod-stt4.log | head -1) (attempt $i)"
      cp pod-stt4.log "pod-stt4-attempt$i-$tag.log"; continue
    fi
    echo "$(date -u +%H:%M) round finished on $tag: $(cat POD-STT4-DONE 2>/dev/null | tr '\n' ' ')"; exit 0
  done <<< "$CARDS"
  sleep 600
done
echo "gave up"
