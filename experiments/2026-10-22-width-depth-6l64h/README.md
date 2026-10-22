# Model size 4L/32H → 6L/64H, polar dense, 36 h on one GPU

**Date.** 2026-10-22. **Status.** running (pod 06po4numi1lii3, community RTX
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

(pending: curve, final bpb, CORE smoke and standard, compute.json)

## Verdict

(pending)

## What it changes

(pending)
