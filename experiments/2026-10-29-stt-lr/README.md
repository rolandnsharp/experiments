# Speech recognition: our block at round 3's learning rate and warmup

**Date.** 2026-10-29. **Status.** concluded, **refuted**, [EXP-0050](https://experiments.blah.dev/EXP-0050),
pre-registered 09:28 UTC before launch. One A40, 9.3 pod-hours (about 3 idle on a slow data download), $5.50.

## Question

EXP-0048's own block with AdamW (peak 1.5e-3, warmup 1,000) learned about half as fast per update as
round 3, whose optimizer ran at 3e-3 with a 200-update warmup. Is the lag the learning rate and warmup?

## Prediction

Whole test-clean greedy WER 0.36–0.42, and at least 0.03 better than EXP-0048.

## Method

EXP-0048 exactly (`harness/stt2`, 5.73 M parameters, 30,000 updates of 240 s, LibriSpeech 960 h, seed 0)
except `--lr 3e-3 --warmup 200` (`harness/stt2/launch-lr.sh`). Scoring: `tune-model3.sh` (format 3).

## Results

| whole sets | 3e-3 / 200 | EXP-0048 (1.5e-3 / 1,000) | round 3 |
|---|---|---|---|
| test-clean, greedy | **0.491** (CER 0.163) | 0.509 | 0.386 |
| test-other, greedy | 0.698 (CER 0.300) | 0.711 | 0.621 |
| Kokoro, greedy | 0.454 (CER 0.141) | 0.468 | 0.370 |
| test-clean, int8 C port, greedy | 0.502 (CER 0.166) | 0.509 | |
| test-clean, int8, beam 64 + 3-gram | **0.259** (CER 0.116) | 0.282 | 0.194 |

alpha 0.5, beta 1.0 chosen on dev-clean 300 (dev WER 0.289); beam 32: 0.276.

Monitoring subset (200 test-clean), this run / EXP-0048 / round 3: 999 1.063 / 1.095 / 1.065;
1,999 0.971 / 1.064 / 0.885; 2,999 0.898 / 0.976 / 0.787; 7,999 0.716 / 0.741 / 0.562;
14,999 0.572 / 0.595 / 0.434; 29,999 0.436.

## Verdict

**Refuted** (both predictions). The higher rate helped early and gained only 0.02 by the end; round 3
stayed 0.10–0.15 ahead throughout. The update rule was the cause: [EXP-0051](../2026-10-29-stt-muon/)
(Muon, same rate) reached 0.338. This run is EXP-0051's control. Int8 cost 0.011 here.
