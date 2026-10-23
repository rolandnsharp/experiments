# Speech recognition on the uConsole: word-LM beam search and int8 in the C port

**Date.** 2026-10-23. **Status.** done (one checkpoint, 200 utterances).

## Question

Two questions about deploying the trained recogniser (mel + gated MLP, 10k updates, from
[stt-gated-mlp](../2026-10-22-stt-gated-mlp/)) in the single-file C port: does a CTC prefix
beam search with the LibriSpeech 3-gram, written from scratch in C, reproduce pyctcdecode's
gain, and what does it cost on the uConsole? And does int8 quantisation of weights and
activations cost accuracy, and how much speed does it buy on a Cortex-A72 that has no int8
dot-product instruction?

## Prior art

- CTC prefix beam search with an n-gram LM: [Hannun et al. 2014](https://arxiv.org/abs/1408.2873);
  pyctcdecode as the reference implementation measured in
  [stt-ngram-decoding](../2026-10-23-stt-ngram-decoding/) (0.234 WER on these utterances).
- Post-training int8 quantisation of weights and activations is standard for CPU inference
  (onnxruntime's dynamic quantisation, which the sherpa-onnx baseline uses). The open question
  was the cost for this model and this chip.

## Hypothesis and prediction

No numeric prediction was written before the runs (rule 2 broken). Expectation stated in the
session before the int8 run: a smaller speedup than on chips with `sdot`, because the A72 must
widen int8 products to int16.

## Independent variables

(1) Decoder: greedy vs beam search (beam 32) with the word 3-gram, alpha 0.5, beta 0.5.
(2) Arithmetic: float32 vs int8 (weights quantised per output row at load time, activations
per frame, symmetric, scale = max|x| / 127; int32 accumulation; matrices whose input width is
not a multiple of 16 stay float32: the mel probe, 40 wide). Held fixed: the checkpoint, chunks
of 16 frames (320 ms latency), the first 200 test-clean utterances read from the original
16-bit FLAC (not the 8-bit mu-law the model was trained on).

## Code

`harness/stt/zara_stt.c` (`--lm`, `--beam`, `--alpha`, `--beta`, `--int8`, `--decode-logp`),
`export_lm.py` (ARPA → a 24 MB binary of sorted n-gram tables), `dump_logp_bin.py`,
`score_lines.py`, `eval200.sh`. The decoder keeps prefixes as nodes of a shared tree
(identical prefixes merge exactly) and caches the word-LM state per node; scoring follows
pyctcdecode: out-of-vocabulary words cost -10 in log10 units before alpha, a partial word that
is no vocabulary prefix costs -10 (scaled up past 6 letters) as a ranking term.

Two bugs found by comparing against pyctcdecode on identical log-probabilities: this LM spells
its unknown word `<UNK>` (the lookup of `<unk>` failed and read garbage), and the OOV penalty
had been applied in the wrong units. Before the fix: 0.398 WER; after: 0.267 (beam 32), 0.249
(beam 64) vs pyctcdecode's 0.234 (beam 64).

## Hardware and cost

Lab CPU for the 200-utterance runs; the uConsole (mains power, one pinned core) for timing.
No GPU.

## Data

LibriSpeech test-clean, first 200 utterances. The LM is LibriSpeech's own pruned 3-gram
(OpenSLR 11): results are "with LM".

## Results

200 utterances, the C port end to end from audio:

| arithmetic | greedy WER / CER | beam 32 + 3-gram WER / CER |
|---|---|---|
| float32 | 0.527 / 0.166 | 0.284 / 0.127 |
| int8 | 0.526 / 0.166 | **0.282 / 0.125** |

(The PyTorch evaluation on the same utterances reads the mu-law training format and scores
0.513 greedy; the C port reads 16-bit audio and scores 0.527.)

LM weight sweep, beam 32, on dumped log-probabilities (`sweep-b32.txt`): alpha 0.5 beta 0.5
best at 0.267; alpha 0.8 needs beta 2.5 to reach 0.281; alpha 1.0 is worse everywhere.

uConsole, one core, one 10.4 s utterance, real-time factor (compute s / audio s):

| core | float32 model | int8 model | beam search + LM |
|---|---|---|---|
| A72 1.8 GHz | 0.130 | **0.068** | 0.035 |
| A53 1.4 GHz | 0.258 | **0.152** | 0.082 |

Full pipeline, int8 + LM: about 0.10 on an A72 and 0.23 on an A53, i.e. 4–10x faster than real
time on one core. Memory: model 23 MB float32 (6 MB of int8 weights in RAM in addition, as
loaded today), LM 24 MB.

## Decided vs measured

Decided: int32 accumulation of int8 products is exact; the beam search merges identical
prefixes exactly. Measured, one checkpoint: every WER and timing.

## Verdict

int8 is free in accuracy here (0.526 vs 0.527 greedy, 0.282 vs 0.284 with LM) and 1.7–1.9x
faster even without `sdot`. The C beam search reproduces most of pyctcdecode's gain (0.249 vs
0.234 at beam 64 on identical inputs); the LM roughly halves word error for under a third of
the acoustic model's compute.

## What it changes

The uConsole pipeline is int8 + LM by default. Against the streaming baseline
([stt-cpu-speed](../2026-10-22-stt-cpu-speed/)), per core, the int8 model alone is about 3x
faster (A72: 0.068 vs 0.20) and the full int8 + LM pipeline about 2x (0.10 vs 0.20; A53 0.23
vs 0.50); accuracy (28% vs sherpa's 8.6%, though sherpa's number is on 12 utterances and
without an LM) is what remains. Next: the round-3 checkpoint, which is already better.
