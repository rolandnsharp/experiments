#!/usr/bin/env bash
# Serve a Vidya checkpoint behind core.roland.blah.dev, register the model on
# evals.blah.dev (once; later runs reuse the id in ~/field/evals-model-id) and
# start a CORE attempt. Run on the lab:
#   bash ~/field/register-vidya.sh RUN_DIR [tier] [label]
# RUN_DIR holds result.json, params.pt and compute.json (compute_cost.py).
set -euo pipefail
cd ~/field
RUN=${1:?run dir}
TIER=${2:-smoke}
LABEL=${3:-}
KEY=$(cat ~/.config/blah/evals-key)
API=https://evals.blah.dev/api/v1
URL=https://core.roland.blah.dev
python3 compute_cost.py "$RUN" > /dev/null
read -r VARIANT MIXING LAYERS HARMONICS PARAMS UPDATES BATCH BPB SECONDS MAC MATRIX RECUR FLOP SEEN < <(python3 - "$RUN" <<'PY'
import json, sys
from pathlib import Path
r = json.loads((Path(sys.argv[1]) / 'result.json').read_text())
c = json.loads((Path(sys.argv[1]) / 'compute.json').read_text())
m = c['macPerByte']
print(r['variant'], r.get('mixing', 'dense'), r['layers'], r['harmonics'], r['params'], r['updates'], r['batch'], f"{r['heldoutBpb']:.4f}", int(r['seconds']),
      f"{m['total']/1e6:.2f}", f"{c['matrixShare']*100:.0f}", f"{m['recurrence']/m['total']*100:.0f}", f"{c['training']['flop']:.2e}", f"{c['training']['bytesSeen']/1e6:.0f}")
PY
)
REV=$(python3 -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:12])" "$RUN/params.pt")
DEVICE=$(grep -m1 -o "device [a-z]*" "$RUN/train.log" | cut -d" " -f2)
echo "== $RUN: $VARIANT/$MIXING, $LAYERS layers x $HARMONICS harmonics, $PARAMS params, $UPDATES updates, held-out $BPB bpb, $MAC M MAC/byte"

# the server: stop an old one (by pid file), start this checkpoint
if [ -f core-server.pid ] && kill -0 "$(cat core-server.pid)" 2>/dev/null; then kill "$(cat core-server.pid)"; sleep 2; fi
OMP_NUM_THREADS=4 nohup ~/zara/.venv/bin/python core_server.py --run "$RUN" --device cpu --port 7990 > core-server.log 2>&1 < /dev/null &
echo $! > core-server.pid
sleep 8
HEALTH=$(curl -s --max-time 30 "$URL/healthz")
echo "== healthz: $HEALTH"
echo "$HEALTH" | grep -q "blah-core-http/1" || { echo "the endpoint is not answering through Caddy"; exit 1; }
curl -s --max-time 120 -X POST "$URL/v1/score" -H 'Content-Type: application/json' \
  -d '{"examples":[{"kind":"choice_set","prompts":["Q: What is 2+2?\nAnswer: 4","Q: What is 2+2?\nAnswer: banana"]}]}'
echo

DESC="Vidya: a byte-level language model whose recurrence is a bank of complex poles (Diakosmos's recurrence with the real decay replaced by a learned complex pole, polar form c = exp(-gamma + i omega)), no attention matrix, no learned embedding. This checkpoint: $VARIANT pole, $MIXING projections, $LAYERS layers x $HARMONICS harmonics (7 heads x 8 channels), $PARAMS parameters, trained $UPDATES updates of batch $BATCH x 2048 bytes ($SEEN MB of OpenWebText bytes, the eval bundle never seen) on $DEVICE in $SECONDS s; held-out next-byte loss $BPB bits per byte on Diakosmos's 4,096 held-out OpenWebText windows (GPT-2 124M: 1.4417, Diakosmos 100M: 1.4194). Compute: $MAC M multiply-adds per byte of context for one forward ($MATRIX% in matrix products, $RECUR% in the recurrence), training about $FLOP FLOP; GPT-2 124M spends 32 M multiply-adds per byte. Scoring runs on 4 CPU threads; a byte is a token, BOS is byte 10, context 2048 bytes. Semantics in Alpha (Vidya.Pole/Cell/Head/Block/Readout, checked against native f32 and Float32Model); this checkpoint from the PyTorch harness. Checkpoint $REV."
if [ -f evals-model-id ]; then
  ID=$(cat evals-model-id)
  curl -s --max-time 30 -X PATCH "$API/models/$ID" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"description": sys.argv[1]}))' "$DESC")" > /dev/null
  echo "== model $ID updated"
else
  R=$(curl -s --max-time 30 -X POST "$API/models" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"name": "Vidya", "description": sys.argv[1], "inference_uri": sys.argv[2], "is_official": False}))' "$DESC" "$URL")")
  echo "$R" | head -c 300; echo
  ID=$(echo "$R" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))")
  [ -n "$ID" ] || { echo "registration failed"; exit 1; }
  echo "$ID" > evals-model-id
  echo "== model registered: $ID"
fi
LABEL=${LABEL:-"$RUN: $VARIANT $MIXING ${LAYERS}L/${HARMONICS}H, $UPDATES updates, $BPB bpb, $MAC M MAC/byte, $FLOP FLOP"}
A=$(curl -s --max-time 60 -X POST "$API/models/$ID/core/attempts" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"score_uri": sys.argv[1], "tier": sys.argv[2], "model_label": sys.argv[3]}))' "$URL" "$TIER" "$LABEL")")
echo "$A" | head -c 400; echo
echo "$A" > "attempt-$(date -u +%Y%m%d-%H%M).json"
