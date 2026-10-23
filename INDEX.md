# Index

## Headline results

The few results that change what we believe, each with how far it is
settled. A row is promoted to *established* only on two seeds or a full run;
*provisional* means one seed or a partial run, and the number can move. The
full record, wins and nulls alike, is the dated table below.

| result | evidence | status |
|---|---|---|
| **The complex (polar) pole beats real decay at equal compute**: 0.036–0.043 bpb at 3.4 M (steady over a 30K pass, unchanged at a 2x tuned rate); **0.103 at 100 M after 640 of 15,360 updates** (1.8794 vs 1.9829, Diakosmos's recipe). Early in training, so not yet evidence that the gain grows with size: at 3.4 M the gap was 0.064 at 1,000 updates and settled at 0.036; the full 100 M run decides. The parametrization is LRU's; the equal-compute measurement is ours | [text-pole-form](experiments/2026-10-21-text-pole-form/), [diakosmos-size-pole](experiments/2026-10-23-diakosmos-size-pole/), [learning-rate-4l32h](experiments/2026-10-23-learning-rate-4l32h/) | 3.4 M **established** (two seeds); 100 M **provisional** (640 of 15,360 updates, one seed) |
| **A twice-width ternary Vidya beats full precision at fewer bytes**: 1.9818 bpb at 3.9 MB vs 2.0157 at 6.77 MB, with 96% of its multiply-adds as ternary adds | [ternary-projections](experiments/2026-10-23-ternary-projections/) | **provisional** (seed 1 and float at 4x pending) |
| **Diakosmos's learning rate (0.003) is too low at 4L/32H**: 2x gains 0.033 bpb for both pole forms | [learning-rate-4l32h](experiments/2026-10-23-learning-rate-4l32h/) | **provisional** (one seed, 3K updates only) |

## All experiments

| date | experiment | variable | result | verdict |
|---|---|---|---|---|
| 2026-10-19 | [speech-pole-form](experiments/2026-10-19-speech-pole-form/) | pole form (real / polar / cartesian) on speech bytes | 3.396 / 3.293 / 3.281 bpb | complex poles win 0.10–0.12 bpb at equal updates (one seed) |
| 2026-10-21 | [text-pole-form](experiments/2026-10-21-text-pole-form/) | pole form (real / polar) on OpenWebText bytes | seed 0: 2.0849 / 2.0490; seed 1: 2.0842 / 2.0409; 30K seed 0: 1.8841 / 1.8469 | polar wins 0.036–0.043 bpb at 3K updates (two seeds) and 0.037 at a full pass (seed 0); the gap does not close with training |
| 2026-10-21 | [grouped-vs-dense-mixing](experiments/2026-10-21-grouped-vs-dense-mixing/) | per-head (block-diagonal) vs dense q/k/v/out projections at equal width | 2.7044 vs 2.0490 bpb | grouped loses 0.65 bpb at equal width (5.3x fewer params); equal-compute comparison untested |
| 2026-10-21 | [training-bytes-vs-bpb-and-core](experiments/2026-10-21-training-bytes-vs-bpb-and-core/) | bytes seen (98 MB → 983 MB, one pass) at 4L/32H | 2.0490 → 1.8469 bpb; CORE smoke 0.004 → 0.044, standard 0.009 → 0.018 | still falling at the end of the pass; data is the cheap lever |
| 2026-10-22 | [width-depth-6l64h](experiments/2026-10-22-width-depth-6l64h/) | model size 4L/32H → 6L/64H (3.4 M → ~20 M params) | running: 69,154 updates = 2.27 GB at 1.72 s/update | prediction: 1.60–1.70 bpb, smoke CORE 0.05–0.06 |
| 2026-10-20 | [speech-long-poles](experiments/2026-10-20-speech-long-poles/) | continuous-time pole, decays 0.1 ms – 2 s, window 16,384 (window and batch also moved) | 3.280 bpb vs real 3.390 / polar 3.285 / cartesian 3.271 | long poles survive training (1.0–1.2 s in every block); no bpb gain (one seed) |
| 2026-10-20 | [stt-front-osc-vs-mel](experiments/2026-10-21-stt-front-osc-vs-mel/) | recogniser front: oscillator bank vs log-mel, Diakosmos blocks, 100 h | mel 0.860 test-clean WER; osc 0.942 vs mel 0.874 at step 3K, then crashed | inconclusive on the front; the backbone plateaus near 85% |
| 2026-10-22 | [stt-gated-mlp](experiments/2026-10-22-stt-gated-mlp/) | gated MLP after every block (schedule length also moved: 10K vs 30K), mel, 960 h | test-clean 0.562 (MLP, 10K) vs 0.757 (plain, 30K) | the missing nonlinearity was the bottleneck, prediction met (one seed) |
| 2026-10-23 | [stt-ngram-decoding](experiments/2026-10-23-stt-ngram-decoding/) | greedy vs beam 64 + LibriSpeech 3-gram | 0.513 → 0.234 WER on 200 test-clean utterances | LM halves the error; mostly spelling (tuned on the reported set) |
| 2026-10-22 | [stt-cpu-speed](experiments/2026-10-22-stt-cpu-speed/) | recogniser on the uConsole CPU: our C port vs whisper.cpp | one A72 core over 87 s: sherpa Zipformer 20M int8 RTF 0.20 at 8.6% WER; ours (MLP, chunked NEON) 0.15 at 49%; whisper tiny 2.26 | after NEON, 1.3–1.4x faster than the streaming baseline per core; far less accurate |
| 2026-10-23 | [stt-front-osc-vs-mel-mlp](experiments/2026-10-23-stt-front-osc-vs-mel-mlp/) | recogniser front: osc vs mel on the MLP backbone, 30K updates | test-clean osc 0.386 vs mel 0.402; test-other 0.621 vs 0.644 | prediction wrong: osc better on all three sets (one seed); ship the oscillator front |
| 2026-10-23 | [diakosmos-size-pole](experiments/2026-10-23-diakosmos-size-pole/) | pole form (real / polar) at Diakosmos's size and recipe (100 M params, 64 windows an update, bfloat16 products) | queued: 640-update check of both arms | prediction: harness real arm 1.96–2.00 at 640 (his 1.9806); polar 0.03–0.08 lower; full run 1.38–1.41 vs his 1.4194
| 2026-10-23 | [stt-c-port-lm-int8](experiments/2026-10-23-stt-c-port-lm-int8/) | C port: greedy vs 3-gram beam search; float32 vs int8 | 200 utt: 0.527 → 0.284 with LM; int8 0.526 / 0.282; A72 RTF 0.130 → 0.068 | int8 free in accuracy, 1.7–1.9x faster; C beam search within 1.5 pts of pyctcdecode |
| 2026-10-23 | [ternary-projections](experiments/2026-10-23-ternary-projections/) | block projections float32 → ternary (BitNet b1.58), same width and twice the harmonics, 4L polar at 3K | same width lr 1x/2x/4x: 2.1280 / 2.0916 / 2.0806; float 2x 2.0157; twice the width (13.2 M, 3.9 MB) at 4x: 1.9818 | prediction: same width 2.15–2.30 bpb (control 2.0490); twice the width within ±0.05 of the control at ~2.5x fewer model bytes |
| 2026-10-23 | [stt-longer-augmented](experiments/2026-10-23-stt-longer-augmented/) | +60K updates from the round-3 osc checkpoint, with augmentation (both move) | planned | prediction: test-clean 0.25–0.30 greedy, 0.11–0.13 with LM |
| 2026-10-23 | [learning-rate-4l32h](experiments/2026-10-23-learning-rate-4l32h/) | peak learning rate 1x / 2x / 4x of Diakosmos's 0.003, real and polar, 4L/32H at 3K | polar 2x 2.0157 (1x 2.0490); real 2x, real 4x, polar 4x running | prediction: real gains at least as much; pole gain at each form's best rate 0.01–0.04 bpb |
| 2026-10-23 | [ternary-wide-30k](experiments/2026-10-23-ternary-wide-30k/) | training length 3K → 30K, wide ternary (4L/64H) vs float 4L/32H, lr 2x | planned (launches if ternary-projections C holds at seed 1) | prediction: the ternary lead grows from ~0.03 to 0.04–0.08 bpb at 30K |
