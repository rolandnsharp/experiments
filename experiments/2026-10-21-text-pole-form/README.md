# Pole form on OpenWebText bytes: real decay vs polar complex pole

**Date.** 2026-10-21 (seed 0, secure 3090), 2026-10-21/22 (seed 1 and the long
real arm, community 3090). **Status.** done at 3K updates with two seeds; done at
30K updates with seed 0 (2026-10-23); the 30K real arm's seed 1 is queued.

## Question

Does the speech result carry to text: does the polar complex pole lower
held-out bpb on OpenWebText bytes at equal size and updates?

## Prior art

- Mamba's ablation: real vs complex diagonals about equal on text, complex
  S4D-Lin initialization worse once selective ([summary](https://sia.hackernoon.com/how-selective-state-space-models-boost-mambas-performance)).
  Our recurrence is not selective and is byte-level, so the question stayed open.
- Parity and modular counting need negative or complex eigenvalues
  ([Grazzi et al. 2025](https://arxiv.org/pdf/2508.07395)) — a capability
  argument, not a bpb one.
- The speech experiment in this repo (2026-10-19): 0.10–0.12 bpb gain.

## Hypothesis and prediction

Written before the run: a smaller gain than on speech, since text has fewer
resonances; expected 0.02–0.05 bpb at 3K updates, and the gain to shrink with
scale.

## Independent variable

The pole form: `real` vs `polar`. Held fixed: 4 layers, 32 harmonics (width
448), dense projections, batch 16 x 2048, 3000 updates, eval every 500 on 256
held-out windows, Diakosmos's optimizer and schedule, the same shard order.
Seeds 0 and 1 for both arms. The polar arm has 14,724 more parameters (the
query's imaginary half) and 0.8% more MAC/byte.

## Code

`harness/field_poles.py train --variant real|polar --mixing dense --layers 4
--harmonics 32 --batch 16 --updates 3000 --seed S`, with `zara_poles.py` and
`diakosmos_oracle.py` as in `harness/`. The difference between arms is the
pole (`zara_poles.pole`, `make_params`) and the doubled state in the
recurrence.

## Hardware and cost

Seed 0: RunPod secure RTX 3090 ($0.50/h, pod xec1bmltj6rml9), 33 + 40 min ≈
$0.61. Seed 1: community RTX 3090 ($0.22/h, pod 9mx8x9l25oj905), 31 + 39 min
≈ $0.26. The 30K real arm: 15,700 updates on the same pod before the stop,
≈ $0.85, then discarded (the old harness had no resume).

## Data

OpenWebText shards 0–1 as bytes (`harness/prepare_owt_bytes.py`;
1,001,644,862 bytes, 200,237 documents, `<|endoftext|>` after each), bytes
seen 98,304,000 per 3K arm. Held-out: the GPT-2 baseline's split reproduced
(18,538,212 scored bytes; held-out input sha256
d068497406362f161eac476b733bca5e0fd525f7e1f8f896bb0fe546eb5ae5ba matches the
published baseline). The CORE eval bundle is never in training.

## Results

Held-out bpb over all 18.5 M scored bytes after 3000 updates:

| seed | real | polar | gap |
|---|---|---|---|
| 0 | 2.0849 | 2.0490 | 0.036 |
| 1 | 2.0842 | 2.0409 | 0.043 |

Curve on the 256-window subset (real / polar, seed 0): 500: 2.677 / 2.447;
1000: 2.285 / 2.221; 1500: 2.168 / 2.119; 2500: 2.087 / 2.048; 3000: 2.059 /
2.028. Seed 1 within 0.01 of seed 0 at every point.

Longer: the polar 30K run (see training-bytes-vs-bpb-and-core) against the
real 30K arm's partial curve, same subset, seed 0: at 13,500 updates real
1.9392 vs polar 1.9021, gap 0.037. (The real arm's `checkpoint.json` holds its
curve to 13,500; `params.pt` at that step is on the lab.)

At a full pass (30,000 updates, 983 MB, one pass over shards 0–1), seed 0,
the real arm rerun from the start on a community 3090 (pod drvotwvc0wxn87,
5.4 h, `real-dense-4L32H-s0-u30000-b16/`), held-out over all 18.5 M scored
bytes:

| updates | real | polar | gap |
|---|---|---|---|
| 30,000 | 1.8841 | 1.8469 | **0.037** |

The subset curve at the same updates (real / polar / gap): 3,000: 2.1000 /
2.0677 / 0.032; 9,000: 1.9689 / 1.9329 / 0.036; 15,000: 1.9256 / 1.8898 /
0.036; 21,000: 1.8997 / 1.8625 / 0.037; 27,000: 1.8662 / 1.8277 / 0.039;
30,000: 1.8601 / 1.8244 / 0.036. The gap holds between 0.032 and 0.039 the
whole way rather than closing.

Compute: real 3.60 M MAC/byte, polar 3.62 M; training FLOP 2.12e15 vs
2.14e15 per 3K arm.

Learned poles (block 0, polar): periods 2.3 bytes to 3.8K bytes, median
18–20 bytes; decay lengths 1.5 to 17 bytes, median 3.2. Real: decay lengths
0.9 to 4.6 bytes, median 2.

## Decided vs measured

Decided: the polar pole's modulus is below 1 for every rho and omega, so the
recurrence is stable by construction, as a verified statement on proofs.blah.dev
(https://proofs.blah.dev/theorems/01a16182-70fb-7c74-8bdc-a9f09546bffb, proof
https://proofs.blah.dev/submissions/01a1618b-48a3-78bb-b719-74cfab2a8a60); the
pole's text and its chain rule are held bit for bit by the Alpha laws
(Proof/VidyaCell), and the chain rule's formulas, dc/drho = -e^rho c and
dc/domega = i c, are verified statements too
(https://proofs.blah.dev/theorems/01a1618c-b671-7583-9af6-3c1f5f8a815f,
https://proofs.blah.dev/theorems/01a1618c-b805-7ea2-9a36-705e476fea9c); the
collection is gathered under the idea
https://proofs.blah.dev/ideas/01a1618f-e0f0-70a8-928d-9766d099a067. The library on proofs.blah.dev already holds verified
results on this recurrence (damped-rotation-recurrence-is-causal-convolution,
damped-rotation-bounded-input-bounded-output, the rounding-error bound), whose
hypothesis ‖c‖ < 1 is exactly the pole statement, so Vidya's states are bounded
by M / (1 − ‖c‖) for bounded inputs. Measured: every bpb number, two seeds at 3K.

## Verdict

Prediction held: polar wins 0.036–0.043 bpb at 3K updates, reproduced at a
second seed (the two arms' seed-to-seed spread is under 0.01, so the gap is
real). At a full pass (30K updates) the gap is 0.037 on seed 0: the
expectation, written before the run, that the gain shrinks with training was
wrong at this size; it held steady. Whether it shrinks with model size is
the diakosmos-size-pole experiment's question.

## What it changes

The polar pole is the kernel's recurrence in Alpha (Vidya.Pole, Vidya.Cell,
PoleRecurrenceSM86). This is the headline experiment: the one variable that
separates Vidya from Diakosmos, at equal parameters and equal multiply-adds
per byte, everything else his. Still owed: a second seed of the 30K real arm
(queued), so the 30K gap rests on two seeds as the 3K gap does. The gain
is small; the case for the architecture rests more on its compute profile
than on the pole.
