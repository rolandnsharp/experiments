# Ternary projections in Vidya: same width and twice the width

**Date.** 2026-10-23. **Status.** seed 0 done: B at 1x, 2x, 4x; float at 2x;
C at 4x. C done at two seeds; float polar 4x landed (2.0039).

## Question

What does it cost in held-out bpb to make Vidya's dense projections ternary
({−1, 0, +1} with one scale per matrix, BitNet b1.58) at 3.4 M parameters, and
does doubling the width buy the cost back at fewer model bytes than the
full-precision model?

This is the first step toward an edge model with no multiplications in the
projections: ternary weights turn every product into an add, a subtract or a
skip. Later steps, each its own experiment, would be 8-bit activations,
shift-only decays (r = 1 − 2^−k) and CORDIC pole rotation, then the speed of
an integer C port on the uConsole with all six cores on (A72s at 1.8 GHz; no battery-mode A53-only tests, Roland 2026-10-23: new batteries will run all cores).

## Prior art

- [BitNet b1.58 Reloaded](https://doi.org/10.1007/978-3-031-66705-3_20)
  (Nielsen and Schneider-Kamp 2024): 1.58-bit quantization-aware training on
  small language models of 100 K to 48 M parameters, our range. Ternary small
  LMs reach the full-precision result **when the hidden layers are doubled**;
  small models respond to learning rate and weight decay differently from the
  large-model recipe. Still open for us: their models are transformers; ours
  carries part of its capacity in complex poles that stay full precision, so
  the cost of ternary projections may be smaller or larger.
- [Spectra](https://doi.org/10.48550/arxiv.2407.12327) (Kaushal et al. 2024,
  99 M to 3.9 B): at equal parameters the ternary models catch up with
  float only at 3.9 B; at equal bits they win above about 1 B. Says to compare
  at equal bytes, not equal parameters, and that the equal-parameter gap at
  our size will be large.
- [Spectra 1.1](https://doi.org/10.48550/arxiv.2506.23025) (Vaidhya et al.
  2025): ternary models gain more from training data than from parameters. We
  are data-bound at this size too (training-bytes-vs-bpb-and-core), so the
  3K-update screen may understate the ternary arms; a full pass would tell.
- [When are 1.58 bits enough?](https://doi.org/10.5220/0013382400003890)
  (Nielsen, Galke and Schneider-Kamp 2025): 1.58-bit training on par with
  float in MLPs, GNNs and encoder models, with a hint of a regularizing effect
  in decoders.

## Hypothesis and prediction

Written 2026-10-23 ~06:30 UTC, before any code or run, against the polar
4L/32H control at 3K updates (2.0490 seed 0, 2.0409 seed 1):

- **B, ternary at the same width: 2.15–2.30 bpb**, a loss of 0.10–0.25.
- **C, ternary at twice the harmonics (64): within ±0.05 of the control**,
  at about 2.5x fewer model bytes (≈13 M ternary weights at 1.58 bits against
  3.4 M at 16 bits).
- Learning rate: the best ternary learning rate is 2–4x the control's.

What would change our mind: B within 0.05 of the control (ternary is nearly
free here; skip the width arm's premise), or C still 0.10 or more behind
(doubling the width is not enough for this architecture; the edge model needs
another lever, not more width).

## Independent variable

- **A → B:** the projection weights in the forward pass: float32 → ternary
  (absmean scale per matrix, round to {−1, 0, +1}, straight-through estimator
  in the backward, latent float32 weights in the optimizer, BitNet b1.58).
  Ternary: every dense 2-D weight inside the blocks (`qkv_proj`, `out_proj`
  and any other dense matrix of the block). Full precision: the byte embedding
  table, the read-out, the norms and every pole parameter (`pole_decay_rho`,
  `pole_omega`, `q_harmonic_weight`).
- **B → C:** harmonics 32 → 64 (width doubles), everything else as B.
- **Also moved, stated:** the learning rate. B is run at the control's rate
  and at 2x and 4x on seed 0; the best rate is used for C and for seed 1 of
  both. So the variable is "ternary with its own learning rate". The 1x arm
  is reported as well, so the cost at the control's rate is visible.
- Held fixed: polar pole, 4 layers, dense mixing, batch 16 x 2048, 3000
  updates, Diakosmos's optimizer and schedule apart from the learning rate,
  the same shard order, eval every 500 on 256 held-out windows, the final
  score over all held-out bytes. Activations stay float32 in this experiment.
  Muon on ternary latent weights is untried in the literature we found;
  if it diverges, the diary says so and the fallback is recorded.

Arm A is not rerun: the text-pole-form polar runs at 3K, seeds 0 and 1, are
the control.

## Code

`harness/field_poles.py train --variant polar --mixing dense --layers 4
--harmonics 32|64 --batch 16 --updates 3000 --seed S --shards 2 --ternary
--lr-scale 1|2|4`. `--ternary` sets `zara_poles.TERNARY`: the forward (the
chunked one and the Sampler) takes every block's `qkv_proj.weight` and
`out_proj.weight` through `zara_poles.ternary` (absmean scale per matrix,
round, clip to {−1, 0, +1}, straight-through gradient); the optimizer is
unchanged and updates the latent float weights (Muon for `out_proj`, the
rotational AdamW for `qkv_proj`). `--lr-scale S` multiplies the oracle's
`PEAK_LR`: Adam's and Muon's rates scale by S, the rotational step's by
sqrt(S) (its eta is sqrt(2 lr wd (1 − beta1) / (1 + beta1))), so `qkv_proj`'s
latent weights move sqrt(S) faster, not S. Without `--half`, Muon on
`out_proj` uses the float32 Newton-Schulz. Both flags default off.

The dense matrices in a block are exactly these two (the harmonic weights are
per-head [H][L][dh] tensors and stay float with the poles), and the read-out
`head.weight` stays float, as BitNet keeps its embedding and head.

`harness/ternary_check.py OLD_DIR`, run on the lab's CPU against the harness
before the change (2026-10-23 06:05 UTC): (1) flag off, the loss (5.677730)
and all 31 gradients on a fixed batch equal the old harness bit for bit;
(2) flag on, the 4 projections take three values times their scale (24.9%
zeros at init), the gradient reaches the latent weights, the loss moves
(5.685822); (3) flag on, the Sampler's logits equal the chunked forward's
(max |diff| 1.3e-6 over 64 positions). A 4-update CPU smoke of `train
--ternary --lr-scale 2` ran end to end. Pods: `pod-self.sh` on Roland's
account, one launcher per rate (`MARKS=ternary-lrS`, `TAG=tern-lrS`).

## Hardware and cost

Roland's own RunPod account (balance $21.47 at 06:10 UTC, with an A40 STT
pod at $0.59/h on it), community RTX 3090s at $0.22/h, the three B rates in
parallel. B lr 1x: pod rm1oe4h6vyyx4s from 06:12 UTC. Estimate: each B arm
≈ 75 min with the fetch and the full held-out score; C ≈ 2.5 h; seed 1 only
if the seed-0 result is worth confirming. About $1.50 for seed 0.

## Data

As text-pole-form: OpenWebText shards 0–1 as bytes, 98,304,000 bytes seen per
3K arm; held-out the GPT-2 baseline's split (sha256 d0684974…, 18,538,212
scored bytes). The CORE bundle is never in training.

## Results

**B, same width, lr 1x, seed 0** (pod rm1oe4h6vyyx4s, 06:12–06:53 UTC):
held-out over all 18,538,212 scored bytes **2.1280 bpb**, against the
control's 2.0490: **+0.079**, a smaller loss than the predicted 0.10–0.25.
Curve on the 256-window subset, ternary / control (text-pole-form polar
seed 0) / gap: 500: 2.5515 / 2.447 / 0.10; 1000: 2.3195 / 2.221 / 0.10;
1500: 2.2114 / 2.119 / 0.09; 2500: 2.1286 / 2.048 / 0.08; 3000: 2.1036 /
2.028 / 0.076. The gap narrows through the run.

3,385,688 parameters: 3,211,264 ternary (95%), 174,424 float. Model bytes:
13.54 MB all float32, 6.77 MB all bfloat16; **1.15 MB** with the ternary
weights packed at 2 bits and the rest bfloat16, 0.99 MB at 1.58 bits (5.9x
under bfloat16). Zeros at the end: 26–27% of every qkv matrix, 33–39% of
the out matrices (39% in the last block), about a third each of +1 and −1.
**B, same width, lr 2x, seed 0** (pod 1mb2d1kisjilnq, 06:29–07:10 UTC):
**2.0916 bpb**, against the control's 2.0490: **+0.043**, inside the ±0.05
band where we said ternary is nearly free. Subset curve, ternary / control /
gap: 500: 2.4315 / 2.447 / −0.02 (ahead early); 1000: 2.2677 / 2.221 / 0.05;
1500: 2.1736 / 2.119 / 0.05; 2000: 2.1348; 2500: 2.0958 / 2.048 / 0.05;
3000: 2.0688 / 2.028 / 0.041. Caveat: the control ran at lr 1x only; if the
float model also gains from 2x, part of this closing is the rate, not
ternary (a float arm at 2x answers it).

**B, same width, lr 4x, seed 0** (pod r0s28vx3pe6jc1, 06:52–07:43 UTC):
**2.0806 bpb**: **+0.032** over the control. Subset curve: 500: 2.3802
(0.07 ahead of the control); 1000: 2.2589; 1500: 2.1645; 2000: 2.1280;
2500: 2.0850; 3000: 2.0554 (+0.027). Across rates the gap goes 0.079 →
0.043 → 0.032 at 1x, 2x, 4x: the gain from the rate is shrinking, so 4x is
taken for C (an 8x arm not run, to save credit).

**A at lr 2x (float, the fairness check), seed 0** (pod 87bthpz2cugrsj,
07:30–08:10 UTC): **2.0157 bpb**, 0.033 better than the float control at 1x.
The float model gains from the higher rate as much as the ternary one does,
so most of the closing above was the rate, not ternary. At matched rate
(2x / 2x) ternary costs **+0.076** (2.0916 vs 2.0157), about what it cost at
1x / 1x (+0.079). Against ternary's best (4x, 2.0806) the gap to float 2x is
+0.065; float at 4x not run yet. The control's own rate (Diakosmos's
PEAK_LR 0.003) is too low for 4L/32H at 3K updates.

**C, twice the harmonics (64, width 896), ternary, lr 4x, seed 0** (pod
dxvdasegg78qa9, 07:48–08:50 UTC, 1.16 s an update): held-out over all
scored bytes **1.9818 bpb**. That is −0.067 against the 1x control (2.0490)
and **−0.034 against the best float run so far** (polar 2x, 2.0157).
Subset curve: 500: 2.2908; 1000: 2.1651; 1500: 2.0675; 2000: 2.0315; 2500:
1.9881; 3000: 1.9581 (float 2x at 3000: 1.9937).

13,189,592 parameters: 12,845,056 ternary, 344,536 float. Model bytes:
**3.90 MB** with the ternary weights at 2 bits and the rest bfloat16 (3.23 MB
at 1.58 bits), against 6.77 MB for the float control in bfloat16: 1.7x
smaller (2.1x at 1.58 bits), not the 2.5x the prediction assumed. The float
remainder doubled with the width. Zeros: 27–28% of every qkv matrix, 34–38%
of the out matrices.

Compute is not equal: **13.64 M MAC/byte**, 3.8x the control's 3.62 M
(matrix 96%, recurrence 4%; GPT-2 124M is 2x that), with 96% of them ternary
adds. Training: 8.05e15 FLOP, 3.8x. So C wins at fewer bytes and fewer
true multiplies, not at equal compute: a float model at this width would
cost 26.4 MB in bfloat16 and the same multiply-adds as true multiplies.

**Float polar at 4x** (learning-rate-4l32h's grid, Vidya session, run
`polar-dense-4L32H-s0-u3000-b16-lr4`): **2.0039 bpb**, the best float run.
At matched 4x rates: same-width ternary loses **+0.077** (2.0806); the wide
ternary arm C wins by **−0.022** (1.9818), down from the −0.034 against float
2x.

**C, seed 1** (pod q9u7f4xbqqjh9y, 09:16–10:23 UTC): **1.9832 bpb**,
0.0014 from seed 0's 1.9818. Subset curve within 0.003 of seed 0 at every
check (1500: 2.0696; 2000: 2.0337; 2500: 1.9911). Against float polar 4x
(2.0039, one seed): −0.021 and −0.022 at seeds 0 and 1.

**CORE (smoke tier) for C seed 0**, scored on 4 lab CPU threads through
core.roland.blah.dev (`core_server.py --ternary`), registered on evals.blah.dev
as the separate unofficial model "Vidya ternary" (id _3-GrSlaIwF20j86AtFfrQ,
attempt DAsVEMQCPx79FPNaZU8axQ): **CORE −0.0106**, chance level. Copa 0.54,
arc_easy 0.28, arc_challenge 0.30, hellaswag 0.28, piqa 0.46, winogrande
0.40, generation tasks 0. For scale: the float 4L/32H at 3K scored 0.004 on
smoke (training-bytes-vs-bpb-and-core); at 98 MB of training no model in this
family is above chance, so CORE does not separate them yet.

Compute as the control's: 3.62 M MAC/byte (matrix 93%, recurrence 7%), but
93% of them are now ternary adds; 2.14e15 training FLOP over 98 MB.

Report beside bpb: parameters, model bytes at the arm's precision (ternary
packed at 2 bits and at 1.58 bits; float at 16 and 32), multiply-adds per byte
split into true multiplies and ternary adds, training FLOP, wall-clock, cost.
The fraction of zero weights per matrix at the end.

## Decided vs measured

Decided, as verified statements on proofs.blah.dev (Lean 4.33.1, Mathlib
0df444a3; 2026-10-23):

- A ternary dot product is one multiply by the scale and otherwise adds and
  subtracts: sum_i (s q_i) x_i = s (sum_{q_i=1} x_i − sum_{q_i=−1} x_i) for
  q_i in {−1, 0, 1}
  (https://proofs.blah.dev/theorems/01a167c2-b5a0-78a7-b962-3ce1d4154873,
  proof https://proofs.blah.dev/submissions/01a167c7-8abc-72c0-bdd7-6f0fa963dc16).
  This is what "93% of the multiply-adds are ternary adds" rests on.
- The absmean quantizer s clamp(r, −1, 1), for any nearest rounding r of w/s
  (ties either way, so torch.round's half-to-even too), moves a weight with
  |w| ≤ 3s/2 by at most s/2
  (https://proofs.blah.dev/theorems/01a167c2-b847-70a1-8bd1-cd38ee0bcf77,
  proof https://proofs.blah.dev/submissions/01a167ca-e101-7fc7-9afb-09aacabe24a9),
  and beyond 3s/2 clips with error exactly |w| − s
  (https://proofs.blah.dev/theorems/01a167c2-ba32-7a7d-befe-44adcbc389e9,
  proof https://proofs.blah.dev/submissions/01a167ca-e2ac-752c-af8a-a776c2e2922d).

Checked by `harness/ternary_check.py` but not proved: the flag-off forward
equals the control's bit for bit; the projections the forward uses take three
values times their scale. Counted: parameters and model bytes per arm.
Measured: every bpb number, one seed so far.

## Verdict

C at two seeds (1.9818, 1.9832); B and the float arms at seed 0:

- B (same width) lost less than predicted at the control's rate (+0.079 vs
  0.10–0.25 predicted). At matched rates it loses 0.065–0.076 to float.
  Ternary at this size costs about 0.07 bpb for a 5.9x smaller model.
- C (twice the width) **beat the prediction**: predicted within ±0.05 of the
  1x control; it is 0.067 below it and 0.022 below the best float run
  (polar 4x, 2.0039), at 1.7x fewer bytes. The byte saving is smaller than predicted (the float
  remainder doubled).
- The learning-rate prediction (best ternary rate 2–4x) held: 4x was best
  of the three tried. The float model also wants 2x or more, which the
  experiment did not anticipate (see learning-rate-4l32h).

Caveats: C at two seeds, B and the float comparison at one; 3K updates only; C has 3.8x the multiply-adds (as adds). The float seed-to-seed spread at 1x was 0.008 (text-pole-form), well under the 0.021 lead.

## What it changes

If C holds: the edge line goes ahead at twice the width (8-bit activations
next), and a ternary arm at Diakosmos's size gives the second point on the
size curve. If C fails: find out whether the loss is in qkv or out (ternary
on one at a time) before going further.
