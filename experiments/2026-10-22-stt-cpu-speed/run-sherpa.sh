#!/usr/bin/env bash
# sherpa-onnx streaming Zipformer 20M (int8) on the uConsole: RTF and WER on the 12 bench utterances.
cd ~/sherpa || exit 1
P=$(cat /sys/class/power_supply/*/status | head -1 | tr A-Z a-z)
B=~/zara-stt/bench
for t in 1 4; do
  out=out-t$t-$P; rm -rf $out; mkdir -p $out
  log=log-t$t-$P.txt
  LD_LIBRARY_PATH=$PWD/lib taskset -c 0-3 ./bin/sherpa-onnx --num-threads=$t --tokens=model/tokens.txt \
    --encoder=model/encoder-epoch-99-avg-1.int8.onnx --decoder=model/decoder-epoch-99-avg-1.int8.onnx \
    --joiner=model/joiner-epoch-99-avg-1.int8.onnx --decoding-method=greedy_search $B/*.wav > $log 2>&1
  python3 - "$log" "$out" <<'PY'
import json, re, sys
from pathlib import Path
log, out = sys.argv[1], sys.argv[2]
text = Path(log).read_text()
cur = None
for line in text.splitlines():
    m = re.search(r'([0-9]+-[0-9]+-[0-9]+)\.wav', line)
    if m: cur = m.group(1)
    if line.lstrip().startswith('{') and cur:
        try: Path(out, cur + '.wav.txt').write_text(json.loads(line)['text'])
        except Exception: pass
rt = re.findall(r'Real time factor \(RTF\): [^\n]*', text)
print(rt[-1] if rt else 'no RTF line')
PY
  echo "threads $t power $P: $(python3 $B/wer.py $B/refs.txt $out)"
done | tee sherpa-result-$(cat /sys/class/power_supply/*/status | head -1 | tr A-Z a-z).txt
