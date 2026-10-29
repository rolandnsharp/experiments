#!/usr/bin/env bash
# Runs ON THE POD, detached, so a dropped ssh session cannot stop it. Fetches and packs
# LibriSpeech (every subset in $SUBSETS, skipping ones already packed), then trains each
# front in $FRONTS. Markers: DATA-DONE, RUN-DONE ("<front> exit=N" lines, then "all").
cd /root/zara/stt || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in $SUBSETS; do
  [ -f data/$s.index.json ] && { echo "$s already packed"; continue; }
  echo "== $s $(date -u +%H:%M)"
  curl -sSL https://openslr.elda.org/resources/12/$s.tar.gz | tar xz -C data \
    && python pack_libri.py data/LibriSpeech/$s data/$s "$(nproc)" \
    && rm -rf data/LibriSpeech/$s \
    || { echo "$s failed"; echo "data failed" >> RUN-DONE; echo all >> RUN-DONE; exit 1; }
done
echo "data ok $(date -u +%H:%M)" >> DATA-DONE
df -h /root | tail -1
for f in $FRONTS; do
  python zara_stt.py train --front $f --data data --train "$TRAIN" --test test-clean --extra-test "$EXTRA_TESTS" \
    --mlp "${MLP:-0}" ${AUGMENT:+--augment} ${WORKERS:+--workers $WORKERS} ${SEED:+--seed $SEED} ${INIT:+--init "$INIT"} --out runs/$f --poles 64 --layers "${LAYERS:-4}" --width "${WIDTH:-448}" --states "${STATES:-0}" ${LR:+--lr $LR} ${WARMUP:+--warmup $WARMUP} --batch-seconds "$BATCH_SECONDS" --updates "$UPDATES" \
    --eval-every "$EVAL_EVERY" --eval-utterances 200 --checkpoint > runs-$f.log 2>&1
  echo "$f exit=$?" >> RUN-DONE
done
echo all >> RUN-DONE
