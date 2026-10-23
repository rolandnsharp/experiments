#!/bin/sh
# Live dictation on the uConsole: headset mic (ES8388, stereo 44.1 kHz, voice = L - R) -> the
# oscillator recogniser (int8) -> 3-gram beam search on each pause. Ctrl-C stops.
cd "$(dirname "$0")" || exit 1
arecord -q -D hw:1 -f S16_LE -r 44100 -c 2 -t raw | ./zara_stt osc-mlp448-r3.bin --stream --mic --int8 --chunk 16 --lm librispeech-3gram.zlm --beam 32
