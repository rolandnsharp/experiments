#!/usr/bin/env bash
# whisper.cpp baseline on this uConsole: real-time factor and WER on 12 test-clean utterances.
cd ~/zara-stt/bench || exit 1
rm -f WHISPER-DONE; mv -f whisper-result.txt whisper-result-old.txt 2>/dev/null
THREADS=${THREADS:-6}
SECS=$(cat seconds.txt)
POWER=$(cat /sys/class/power_supply/*/status 2>/dev/null | head -1 | tr A-Z a-z)
echo "cpu online $(cat /sys/devices/system/cpu/online), $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) kHz, power $(cat /sys/class/power_supply/*/status 2>/dev/null | head -1), threads $THREADS, audio $SECS s"
for m in tiny.en base.en; do
  mkdir -p out-$m; rm -f out-$m/*
  t0=$(date +%s.%N)
  ~/src/whisper.cpp/build/bin/whisper-cli -m ~/src/whisper.cpp/models/ggml-$m.bin -t $THREADS -np -nt -otxt *.wav > log-$m.txt 2>&1
  # -of with many files is ignored per file; outputs land next to the inputs as <file>.wav.txt
  mv *.wav.txt out-$m/ 2>/dev/null
  t1=$(date +%s.%N)
  wall=$(python3 -c "print(f'{$t1-$t0:.1f}')")
  rtf=$(python3 -c "print(f'{($t1-$t0)/$SECS:.2f}')")
  echo "$m: wall $wall s for $SECS s audio, real-time factor $rtf (lower is faster; 1.0 = real time)  $(python3 wer.py refs.txt out-$m)"
done | tee whisper-result-$POWER.txt
echo all >> WHISPER-DONE
