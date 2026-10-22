#!/usr/bin/env bash
# Runs ON THE POD, detached: waits for the running mel training to finish, then trains the
# backbone probe (mel front + gated MLP) in place of the osc run. Appends to RUN-DONE like
# pod-run-all.sh so the lab driver fetches and stops the pod as usual.
cd /root/zara/stt || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while ps -eo args | grep -v grep | grep -q "zara_stt.py train --front mel --data data --train train-clean-100,train-clean-360,train-other-500 --test test-clean --extra-test test-other,zara-test --out runs/mel "; do sleep 30; done
echo "mel exit=done" >> RUN-DONE
python zara_stt.py train --front mel --mlp "$MLP" --data data --train train-clean-100,train-clean-360,train-other-500 --test test-clean --extra-test test-other,zara-test \
  --out runs/mel-mlp$MLP --poles 64 --layers 4 --harmonics 32 --batch-seconds 240 --updates "$UPDATES" \
  --eval-every 1000 --eval-utterances 200 --checkpoint > runs-mel-mlp$MLP.log 2>&1
echo "mel-mlp$MLP exit=$?" >> RUN-DONE
echo all >> RUN-DONE
