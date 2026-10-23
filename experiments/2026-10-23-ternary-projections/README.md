# Ternary projections in Vidya: same width and twice the width

**Date.** 2026-10-23. **Status.** running: the same-width arm (B) at the
control's learning rate on a community 3090 since 06:13 UTC; the 2x and 4x
rates waiting for a GPU; the wide arm (C) after them.

## Question

What does it cost in held-out bpb to make Vidya's dense projections ternary
({−1, 0, +1} with one scale per matrix, BitNet b1.58) at 3.4 M parameters, and
does doubling the width buy the cost back at fewer model bytes than the
full-precision model?

This is the first step toward an edge model with no multiplications in the
projections: ternary weights turn every product into an add, a subtract or a
skip. Later steps, each its own experiment, would be 8-bit activations,
shift-only decays (r = 1 − 2^−k) and CORDIC pole rotation, then the speed of
an integer C port on the uConsole's A53 cores.

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

(none yet)

Report beside bpb: parameters, model bytes at the arm's precision (ternary
packed at 2 bits and at 1.58 bits; float at 16 and 32), multiply-adds per byte
split into true multiplies and ternary adds, training FLOP, wall-clock, cost.
The fraction of zero weights per matrix at the end.

## Decided vs measured

Decided (to write as laws): the quantizer's output takes exactly three
values times the scale; the flag-off forward is the control's forward bit for
bit; the parameter and byte counts per arm. Measured: every bpb number.

## Verdict

(none yet)

## What it changes

If C holds: the edge line goes ahead at twice the width (8-bit activations
next), and a ternary arm at Diakosmos's size gives the second point on the
size curve. If C fails: find out whether the loss is in qkv or out (ternary
on one at a time) before going further.
