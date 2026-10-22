# Grouped (per-head, block-diagonal) vs dense q/k/v/out projections at equal width

**Date.** 2026-10-21. **Status.** done (one seed, equal width only).

## Question

The oscillator model's point is fewer matrix products. If each head projects
only its own 64 channels (a block-diagonal restriction of the dense
projection), the projection parameters fall 7x and MAC/byte 4x. How much bpb
does that cost at equal width?

## Prior art

- GroupBERT ([Graphcore](https://www.graphcore.ai/posts/groupbert)): grouped
  (block-diagonal) matrices only in the FFN down-projection, with an output
  projection added back to restore cross-group mixing — grouping alone loses
  information flow.
- Structured Hadamard output projection ([arXiv 2603.08343](https://arxiv.org/pdf/2603.08343)):
  replaces the dense head-mixing projection with a fixed transform plus an
  affine; reports comparable performance with 25% fewer attention parameters
  per block (not independently reproduced).
- No controlled ablation of block-diagonal vs dense q/k/v/out at matched
  budgets was found; the question was open for this model.

## Hypothesis and prediction

Written before the run: grouped loses, by an amount that decides whether the
kernels are written dense or grouped; a loss under 0.1 bpb would make grouped
the default (its compute profile is the whole argument), a loss over 0.3 would
not.

## Independent variable

Mixing: `dense` (Diakosmos's fused [3·448 x 448] projection and [448 x 448]
out) vs `grouped` (per head [3·64 x 64] and [64 x 64]). Held fixed: polar pole,
4 layers, 32 harmonics (width 448), batch 16 x 2048, 3000 updates, seed 0,
optimizer and schedule, data order. Parameters: 3,385,688 vs 633,176 (not
matched: this is the equal-width comparison).

## Code

`harness/field_poles.py --mixing dense|grouped`; the grouped path in
`zara_poles.py` (`qkv_group.weight` [H][3dh][dh], `out_group.weight`
[H][dh][dh], block and sampler paths). `grouped_check.py` (on the lab) holds
grouped equal to the block-diagonal dense product to 1e-15.

## Hardware and cost

RunPod secure RTX 3090 ($0.50/h, pod xec1bmltj6rml9), 40 min each ≈ $0.67.

## Data

As text-pole-form: OpenWebText shards 0–1, 98,304,000 bytes seen, held-out
18,538,212 scored bytes.

## Results

| arm | params | held-out bpb | MAC/byte | matrix share | training FLOP |
|---|---|---|---|---|---|
| dense | 3,385,688 | **2.0490** | 3.62 M | 93% | 2.14e15 |
| grouped | 633,176 | 2.7044 | 0.87 M | 70% | 5.1e14 |

Curve (256 windows, dense / grouped): 500: 2.447 / 3.146; 1000: 2.221 /
2.899; 1500: 2.119 / 2.793; 2500: 2.048 / 2.712; 3000: 2.028 / 2.689. The
grouped curve is still falling at 3K but 0.66 behind at every point.

## Verdict

Grouped loses 0.65 bpb at equal width, far over the 0.3 threshold. The dense
projections hold 80% of the parameters and most of the capacity at this size.
One seed; the gap is many times the seed spread seen in text-pole-form.

## What it changes

Kernels and layout are dense (fused projection at stride 1536, out 512 x 512).
Untested and still worth running: grouped at equal compute (wider heads, more
channels) rather than equal width, since that is the comparison the hardware
argument needs.
