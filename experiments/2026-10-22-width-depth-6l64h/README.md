# Model size 4L/32H → 6L/64H, polar dense, 36 h on one GPU

**Date.** 2026-10-22. **Status.** running (pod 5mve5k8pmmu0c6 after a restart, community RTX
3090, started 01:58 UTC). This section was written before the benchmark
finished; the results section is empty until the run ends.

## Question

Does a 6x larger model (width 896, 6 layers, ~20 M parameters, 20.3 M
MAC/byte, still 1.6x under GPT-2 124M's 32 M) on more bytes reach the
leaderboard's reference points, and how does bpb move per parameter?

## Prior art

- The training-bytes experiment (this repo): 4L/32H still data-bound at 983 MB.
- BLT and the byte-distillation paper (cited there) place byte models on the
  same compute-scaling curve as token models at 1 B+ parameters; nothing says
  where a 20 M byte model with a recurrence instead of attention lands.

## Hypothesis and prediction

Written 2026-10-22 01:00 UTC, before the run: held-out **1.60–1.70 bpb**; smoke
CORE **0.05–0.06** (Ajax's Alpha GPT-2 sits at 0.061); standard CORE
0.03–0.04. If bpb ends above 1.75, the model is still data-bound rather than
size-bound at this scale and the next run should move data alone. If the
benchmark step time makes the 36 h cover under 1.2 GB, the result says more
about size than about data.

## Independent variable

Not clean: model size (4L/32H → 6L/64H, 3.4 M → ~20 M parameters) and bytes
seen (983 MB → whatever 36 h affords) move together. Held fixed: polar pole,
dense mixing, batch 16 x 2048 (halved on OOM; the diary says), seed 0,
optimizer and schedule, shard order (the first N of 10 shards). The clean
follow-ups are listed in the training-bytes paper.

## Code

`harness/field_poles.py` with `--layers 6 --harmonics 64 --shards 10
--updates <from benchmark> --resume`; `harness/pod-self-driver.sh` v2 (auto
budget: a 30-update benchmark sets the count for 36 h at 0.92 duty; resume
from the lab's copy if the host drops). Memory-mapped shards (`ByteShards`),
bit-identical to the previous loader.

## Hardware and cost

Community RTX 3090, $0.22/h (Ajax's rule is 3070 first; none was in stock),
36 h training + scoring ≈ $8.50. Watchdog 40 h.

## Data

The first N of 10 OpenWebText shards (5.0 GB prepared;
`harness/owt-manifest.json`), N set by the update count; one pass, no
repetition. Held-out as before (18,538,212 scored bytes).

## Results

Benchmark (02:18 UTC): **1.72 s an update** at batch 16 on the community 3090
(the 4L/32H model ran at 0.87 s), so the 36 h budget set **69,154 updates =
2.27 GB of bytes** (4.5 shards of the 10, one pass). Run directory on the
lab: `~/field/runs/polar-dense-6L64H-s0-u69154-b16`; the diary pushes a
256-window held-out check every 30 min. Expected end ~14:30 UTC Oct 23.

**Interim, 2026-10-22 ~16:00 UTC (step 16,120 of 69,154; ~30,000 s run
time).** The run restarted once: the first pod (06po4numi1lii3, host
gos29w9nu2j0) died at step 660 with "CUDA error: unspecified launch failure"
in the backward; the driver resumed from the step-500 checkpoint (optimizer
state included) on pod 5mve5k8pmmu0c6 (community 3090). The 256-window
held-out subset every 1,000 updates, beside the 4L/32H 30K run's at the same
updates (same batch, so the same bytes seen; the training-bytes experiment's
`polar-dense-u30000/train.log`):

| updates | MB seen | 4L/32H subset bpb | 6L/64H subset bpb | gap |
|---|---|---|---|---|
| 1,000 | 33 | 2.2242 | 2.0633 | +0.161 |
| 2,000 | 66 | 2.1036 | 1.9416 | +0.162 |
| 3,000 | 98 | 2.0677 | 1.8913 | +0.176 |
| 4,000 | 131 | 2.0138 | 1.8526 | +0.161 |
| 5,000 | 164 | 1.9923 | 1.8266 | +0.166 |
| 6,000 | 197 | 1.9863 | 1.8157 | +0.171 |
| 7,000 | 229 | 1.9552 | 1.7849 | +0.170 |
| 8,000 | 262 | 1.9564 | 1.7818 | +0.175 |
| 9,000 | 295 | 1.9329 | 1.7654 | +0.167 |
| 10,000 | 328 | 1.9241 | 1.7519 | +0.172 |
| 11,000 | 360 | 1.9207 | 1.7501 | +0.171 |
| 12,000 | 393 | 1.9076 | 1.7385 | +0.169 |
| 13,000 | 426 | 1.9173 | 1.7463 | +0.171 |
| 14,000 | 459 | 1.8972 | 1.7210 | +0.176 |
| 15,000 | 491 | 1.8898 | 1.7181 | +0.172 |
| 16,000 | 524 | 1.8872 | 1.7161 | +0.171 |

Read as: at equal bytes the larger model is 0.17 bpb ahead by 16K updates,
and the gap has been widening slowly (0.16 at 1K, 0.17 at 16K) rather than
closing, so the 4L model was size-bound as well as data-bound. The subset
runs about 0.02 below the full held-out at this scale (30K run: subset
1.8238 at 29.5K vs full 1.8469). Not a verdict: the prediction is on the
full held-out at the end of the run, which also sees 2.3x the bytes.

(pending: the full curve, final bpb, CORE smoke and standard, compute.json)

## Verdict

(pending)

## What it changes

(pending)
