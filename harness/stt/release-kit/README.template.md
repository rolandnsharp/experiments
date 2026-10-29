---
license: mit
language: en
library_name: c
pipeline_tag: automatic-speech-recognition
tags:
  - speech-recognition
  - streaming
  - edge
  - state-space
  - oscillator
  - ctc
datasets:
  - openslr/librispeech_asr
metrics:
  - wer
---

# Oscillator streaming speech recogniser ({params_m} M parameters, one C file)

A small English speech recogniser that transcribes **while you speak**, built for a handheld
ARM computer (ClockworkPi uConsole). Text appears every 320 ms instead of after you stop, and the
whole model runs **faster than real time on one slow ARM core**, from a single dependency-free C
file.

It is a research model with honest limits: accurate enough to dictate with on clean speech,
well behind Whisper and production systems on accuracy. The point is the architecture and its
cost.

## How it works

- **Front end: a bank of 64 damped complex oscillators** running at the 16 kHz sample rate,
  z_t = c z_(t-1) + x_t with c = exp(-dt/tau + 2 pi i f dt). Frequencies and decay times are
  learned. No FFT, no convolution. Every 20 ms the bank's state (real, imaginary, energy) is the
  feature vector.
- **Backbone: {blocks_word} recurrent blocks of complex poles** at 50 Hz (decays from 20 ms to seconds),
  each followed by a gated MLP. No attention: the state is a fixed size, so cost and memory per
  second of audio are constant, and output is available as the audio arrives.
- **CTC over characters**, decoded greedily as you speak, and re-decoded at each pause with a
  word 3-gram (prefix beam search, also in the C file).

Every experiment that led here, including the failures, is written up with predictions made before each run ({experiments_link}).

## Results

LibriSpeech, word error rate (lower is better), all 2,620 / 2,939 test utterances. "LM" = beam
search (width 64) with LibriSpeech's own pruned 3-gram (OpenSLR 11), its two weights chosen on
dev-clean.

| model | params | test-clean greedy | test-other greedy | test-clean + LM |
|---|---|---|---|---|
{results_rows}

Speed on the uConsole (ClockworkPi CM4-class board), one core, real-time factor = compute
seconds per second of audio (lower is faster), 87 s of test audio, model and LM loading included:

| recogniser | streaming | A72 1.8 GHz | A53 1.4 GHz | WER on these 12 utterances |
|---|---|---|---|---|
| **this model**, int8 + 3-gram | yes, 320 ms | **{rtf_a72}** | **{rtf_a53}** | {wer12} |
| sherpa-onnx streaming Zipformer 20M, int8 | yes | 0.20 | 0.50 | 8.6 % (no LM) |
| whisper.cpp tiny.en | no (30 s window) | 2.26 (1 thread) | 4.49 (1 thread) | 5.0 % |

## Use

```sh
cc -O3 -ffast-math -o zara_stt zara_stt.c -lm
./zara_stt model.bin speech.wav --int8 --lm librispeech-3gram.zlm --alpha {alpha} --beta {beta}   # 16 kHz mono WAV
arecord -q -f S16_LE -r 16000 -c 1 -t raw | ./zara_stt model.bin --stream --int8 --lm librispeech-3gram.zlm --alpha {alpha} --beta {beta}
```

`--stream` prints letters live and a cleaned-up line after each pause. On the uConsole headset
use `dictate.sh` (the codec delivers the mic as left minus right).

## Training

LibriSpeech 960 h (train-clean-100, -360, -other-500), 16 kHz mu-law, {updates} updates of 240 s
batches (about {passes} passes), one {gpu}, about {train_hours} hours, about ${cost}. Shape: osc front
(64 poles), {blocks} blocks of width 448, gated MLP {mlp}. The language-model weights used by
default (alpha {alpha}, beta {beta}, passed by `dictate.sh`) were chosen on dev-clean.

## Limitations

- English read speech only; accents and microphones outside LibriSpeech are weaker.
- No punctuation or capitals.
- The language model is built from public-domain books: modern and technical words are often
  misrecognised.
- Accuracy far below Whisper and production recognisers.

## Files

`zara_stt.c` (inference: oscillator bank, blocks, int8 NEON kernels, beam search, streaming),
`model.bin`, `params.pt` (PyTorch), `librispeech-3gram.zlm` (the 3-gram in the C file's format),
`zara_stt.py` and `zara_block.py` (training: model, block, optimizer), `mulaw.py`, `pack_libri.py`,
`export_c.py`, `export_lm.py`, `score_lines.py`, `dictate.sh`, `SHA256SUMS`.

## Credits

Roland Sharp, with Claude (Anthropic). Data: LibriSpeech audio (Panayotov et al., 2015), CC BY 4.0 (OpenSLR 12); the 3-gram and vocabulary
are from OpenSLR 11 (public domain).
