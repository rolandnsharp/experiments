# Speech recognition: greedy CTC vs beam search with the LibriSpeech 3-gram

**Date.** 2026-10-23. **Status.** done (one checkpoint, 200 utterances).

## Question

How much of the MLP recogniser's error is spelling that a word language model fixes, with
the acoustic model unchanged?

## Prior art

- CTC prefix beam search with an n-gram LM is standard practice since
  [Graves & Jaitly 2014](https://proceedings.mlr.press/v32/graves14.html) and
  [Hannun et al. 2014](https://arxiv.org/abs/1408.2873); LibriSpeech ships its own n-gram
  LMs ([Panayotov et al. 2015](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf),
  OpenSLR 11). Large relative gains for weak character models are well known; the number for
  this model was not.

## Hypothesis and prediction

No numeric prediction was written before the run (rule 2 broken). The observation that
prompted it: greedy outputs were phonetically close and misspelt ("STOO PERDINER" for
"STEW FOR DINNER").

## Independent variable

The decoder: greedy (best class per frame, collapse, drop blanks) vs pyctcdecode prefix beam
search, beam 64, with the LibriSpeech pruned 3-gram (`3-gram.pruned.3e-7.arpa`) and its
200k-word vocabulary as unigrams, over a 3 × 2 grid of LM weight alpha ∈ {0.3, 0.5, 0.8} and
word bonus beta ∈ {0.5, 1.5}. Held fixed: the MLP checkpoint from
[stt-gated-mlp](../2026-10-22-stt-gated-mlp/) (md5 654109dba9681701c9c3ad3385fd7c9b), its
log-probabilities on the first 200 test-clean utterances.

The grid was tuned on the same 200 utterances it is reported on; with six settings the
optimism is small but nonzero. A clean number needs tuning on dev-clean.

## Code

`harness/stt/dump_logp.py`, `harness/stt/decode_eval.py`; output in `decode.txt`.

## Hardware and cost

Lab CPU (5 threads), a few minutes; no GPU.

## Data

The LM is trained on LibriSpeech's book text (not the test transcripts); it is the in-domain
LM every LibriSpeech result "with LM" uses. Results must be reported as "with LM".

## Results

| decoding | WER | CER |
|---|---|---|
| greedy | 0.513 | 0.160 |
| beam 64 + 3-gram, alpha 0.3 beta 0.5 | 0.267 | 0.104 |
| alpha 0.5 beta 0.5 | **0.234** | **0.098** |
| alpha 0.5 beta 1.5 | 0.243 | 0.098 |
| alpha 0.8 beta 0.5 | 0.254 | 0.110 |
| alpha 0.8 beta 1.5 | 0.234 | 0.102 |

```
ref:    AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS
greedy: AFTER EARLY NIGHT FALL THE ELLOW LAMS WOULD LIGHT HAP PERIN THERE  THE SQUALEED QUARTER OF THE BROFFELS
LM:     AFTER EARLY NIGHT FALL THE YELLOW LAMPS WOULD LIGHT UP HER AND THERE THE SQUALID QUARTER OF THE BROTHELS
```

## Decided vs measured

All measured: one checkpoint, 200 utterances, tuned on the reported set.

## Verdict

The LM more than halves the word error (0.513 → 0.234) and cuts the character error by 40%.
Most greedy errors are orthographic; what remains is mostly genuine mishearing ("STOWER
DINNER", "COUNTED" for "COUNSELLED").

## What it changes

Report both numbers for every recogniser from here, LM tuned on dev-clean. The uConsole
port needs a beam decoder with a small LM to show the "with LM" accuracy live.
