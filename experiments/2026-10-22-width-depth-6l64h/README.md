# Model size 4L/32H → 6L/64H, polar dense, 36 h on one GPU

**Date.** 2026-10-22. **Status.** done (bpb); CORE pending. This section was written before the benchmark
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

**Final (2026-10-24 01:02 UTC).** The run was paused once at update 35,500 for
the real-decay 30K control (2026-10-22 21:17 UTC) and resumed from its
checkpoint on the shared account; 34.9 h of training in all. Held-out over all
18,538,212 scored bytes after 69,154 updates (2.27 GB, one pass):
**1.5799 bpb**. 19,667,012 parameters, 20.34 M MAC/byte (matrix 96%,
recurrence 4%), training 2.77e17 FLOP.

Subset curve (256 windows): 10K 1.7519, 20K 1.6865, 30K 1.6464, 40K 1.6152,
50K 1.5881, 60K 1.5682, 69,154 1.5575 — still falling at the end.

(pending: CORE smoke and standard)

## Verdict

The bpb prediction (1.60–1.70) was too pessimistic: 1.5799, below its lower
end. Not a clean size result, since size and bytes moved together (3.4 M at
983 MB gave 1.8469). Still data-bound: the curve had not flattened at 2.27 GB.
CORE pending.

## What it changes

At 20 M parameters and 2.3 GB Vidya is at 1.58 bpb, against Diakosmos's 1.42 at
100 M and 2.0 GB; the size and recipe comparison now runs as
diakosmos-size-pole.
