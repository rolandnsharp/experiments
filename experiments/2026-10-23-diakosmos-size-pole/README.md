# The polar pole at Diakosmos's size and recipe: against his 1.4194

**Date.** 2026-10-23. **Status.** running: the real arm's 640-update check is done (prediction 1
held); the polar arm's first leg runs on Roland's account.

## Question

Does the polar pole, swapped into Diakosmos at his full size and with his
exact recipe, beat his trained result of 1.4194 bits a byte?

## Prior art

- Diakosmos (Ajax, Alpha, 12–13 October 2026; `website/ops/diakosmos/gates.json`
  in the Alpha repository): 99,823,934 parameters (14 layers, width 1330, 7
  heads, 8 channels a head), 15,360 updates of 64 windows of 2,048 bytes read
  once in order (2.01 GB), bfloat16 product operands with binary32
  accumulation, on one RTX 3070 in 24.5 GPU hours. Held-out bits a byte on the
  GPT-2 baseline's 18,538,212 bytes: 1.9806 at update 640, 1.6842 at 2,560,
  1.5712 at 5,120, 1.4673 at 10,240, **1.4194 ± 0.0023** at the end. GPT-2
  small on the same bytes and card: 1.4417 after 25.2 GPU hours.
- text-pole-form (this repo): the polar pole beats the real decay by
  0.036–0.043 bpb at 3.4 M parameters and 3K updates, two seeds, at equal
  parameters and multiply-adds; the gap was 0.037 at 13.5K updates on one seed.
- width-depth-6l64h (this repo): no pole comparison; a scaling run.

## Hypothesis and prediction

Written 2026-10-23 before any run at this size.

1. **The harness reproduces Diakosmos.** Our real-decay arm, with his
   geometry, initialization, optimizer, schedule (for 15,360 updates), 64
   windows an update and his device's product rounding, scores **1.96–2.00
   bpb** on the full held-out set after 640 updates (his 1.9806). The training
   bytes are the same OpenWebText documents under the same held-out rule, but
   possibly in a different order (his stream was decoded from GPT-2 token
   shards, ours from the parquet files in name order), so an exact match is
   not expected. Outside 1.95–2.01, the harness is not a faithful Diakosmos
   and the head-to-head is called off until the difference is found.
2. **The pole helps early.** The polar arm at 640 updates is **0.03–0.08 bpb
   below** the real arm (at 3.4 M parameters the gap was 0.064 at 1,000
   updates and narrowed to 0.04 by 3,000).
3. **The pole helps at the end.** Run to 15,360 updates, the polar arm ends at
   **1.38–1.41 bpb**, beating 1.4194 by 0.01–0.04. Below a 0.01 gain (inside
   about four of his standard errors) the pole buys nothing at his scale, and
   the case for Vidya rests on compute, not quality.

## Independent variable

The pole form, real decay vs polar complex pole (Vidya.Pole). Held fixed:
Diakosmos's geometry (14 layers, 95 harmonics, width 1330, 7 heads of 190, 8
channels a head), his initialization (`diakosmos_oracle.initialize`, seed 0;
the polar arm's decays start where the real arm's are and its frequencies
log-spaced, as in every pole run here), his optimizer and schedule for 15,360
updates (`diakosmos_oracle.Update`, peak 0.003, warmup 200, cosine to 0.1,
clip 1.0, Muon on the out projections with the bfloat16 Newton-Schulz,
rotational AdamW on the fused projection and the head, Adam on the rest), 64
windows of 2,048 bytes an update (8 batches of 8), the device's product
rounding (`--half`). The polar arm has 99,974,252 parameters (+0.15%: the
query harmonics' imaginary half).

## Code

`harness/field_poles.py` with three new flags, each off by default:
`--half` (the products' operands rounded to bfloat16, multiplied exactly and
accumulated in binary32, as `diakosmos_oracle.HalfProduct`, on tensor cores
through TF32 enabled for those products only; Muon's
`newton_schulz_half`), `--accumulate K` (an update is the mean gradient of K
batches), `--stop-at N` (save, score the full held-out set, stop; `--resume`
continues). Checked on the lab's CPU before use: without the flags a run is
bit for bit the previous harness's (parameters and held-out equal); two
batches of one equal one batch of two to 3e-7; a run stopped at 4 of 8
updates and resumed equals the uninterrupted run bit for bit; the real
variant at this geometry has 99,823,934 parameters, Diakosmos's own count.
`harness/pod-self.sh` and `pod-self-driver.sh` pass `EXTRA` arguments and a
`TAG` for the run name; `harness/chain-v2.sh` queues the check.

The legs: `LEGS="real:dense:15360:0 polar:dense:15360:0" LAYERS=14
HARMONICS=95 BATCH=8 SHARDS=1 EXTRA="--half --accumulate 8 --stop-at 640
--eval-every 640 --eval-windows 64 --final-windows 0" TAG=dk`.

## Hardware and cost

One community RTX 3090 ($0.22/h), the 640-update legs after the 30K control,
on the shared lab account. The full 15,360-update polar run (from the 640
checkpoint, `--resume`) runs on Roland's own RunPod account (Roland,
2026-10-23), started by hand after the 640-update verdict with
`harness/launch-dk-full.sh`, which sizes the run from the 640-update leg's
step time and refuses to start unless the account's balance covers it, the
account's other pods meanwhile, and $3 of margin. Cost not yet known: the
step time sets it. His run took 24.5 hours on a 3070 with his own kernels.

## Data

OpenWebText bytes, `harness/owt-manifest.json` (Skylion007/openwebtext at
79d93d78, held-out rule sha256(text)[:4] % 2000 == 0, as
diakosmos-prepare-bytes.py), the first shard for the 640-update legs (84 MB
read), the first five for a full run (2.01 GB, one pass). Held-out: the GPT-2
baseline's 4,096 windows, 18,538,212 scored bytes (input sha256
d0684974…, as his).

## Results

**Prediction 1 (the harness reproduces Diakosmos): held.** The real-decay
arm, Diakosmos's recipe in our harness (`--half --accumulate 8`, batches of 8,
activation checkpointing, uncompiled), on a community RTX 3090 of the shared
account (pod ffpol6p39er7dp, 03:47–07:54 UTC), after 640 updates of the
15,360-update schedule, over all 18,538,212 held-out bytes:

| | held-out bpb at update 640 |
|---|---|
| Diakosmos (his run, Alpha, RTX 3070) | 1.9806 |
| real decay, this harness | **1.9829** |

The difference, 0.0023, equals his final standard error; the training bytes
may be in a different document order. 20.5 s an update (13,574 s for the
640 updates and the full score).

**Speed.** prof_harness.py (one 64-window update at this size): the harness
is host-bound, the Python loop of the chunked recurrence leaving the GPU
waiting (CPU self time about three times CUDA time). On an RTX 4090: 19.1 s
as run above, 7.9 s with the recurrence compiled (`--compile recurrence`,
checked equal to the uncompiled run up to rounding; compiling the whole block
moved parameters by 21% and is not used). An A100 80GB was slower than the
4090 uncompiled (38.9 s), its host being slower.

**The polar arm** runs on Roland's account on an RTX 4090 with the recurrence
compiled (`launch-dk-full.sh first`, run `polar-dense-14L95H-s0-u15360-b8-dkc`,
from 07:54 UTC): updates 0–640 and the full score, then, if prediction 2
holds, the same run continued to 15,360. Compiled versus uncompiled and 4090
versus 3090 differ at the rounding level only, far below the predicted gap.

(pending: the polar arm at 640)

## Decided vs measured

Decided by the Alpha checker or verified on proofs.blah.dev: the polar pole's
modulus is below 1 for every rho and omega, and its chain rule (see
text-pole-form). Measured here: every bpb number, one seed per arm.

## Verdict

(pending)

## What it changes

If prediction 1 holds and 2 holds, the full polar run is the head-to-head
with Diakosmos and the result to send to Ajax. If 1 fails, the harness is
fixed first.
