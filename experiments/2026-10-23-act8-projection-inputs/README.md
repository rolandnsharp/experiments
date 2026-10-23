# 8-bit projection inputs on the wide ternary Vidya

**Date.** 2026-10-23. **Status.** running: launched 12:13 UTC on Roland's account (MARKS `act8-w64`), waiting for 3090 stock. The code and the CPU check are
done. The prediction was written at 12:10 UTC,
before any run with the flag.

## Question

The wide ternary Vidya (4L/64H, ternary projections, lr 4x) scores 1.9818 and
1.9832 bpb at seeds 0 and 1. What does it cost to also round each
projection's input to int8? With that, every projection is a sum of int8
values with signs: integer adds into an int32 accumulator, plus one float
multiply per output (the weight scale times the input scale).

## Prior art

- [BitNet b1.58](https://arxiv.org/abs/2402.17764) (Ma et al. 2024): ternary
  weights with 8-bit absmax activations per token are the published recipe.
  The full-precision comparisons there are made with 8-bit activations
  already in place, so the activation cost alone is not reported at our
  size.
- [BitNet a4.8](https://arxiv.org/abs/2411.04965) (Wang et al. 2024): 4-bit
  activations need care for outlier channels (sparsification, a hybrid
  scheme). That points at the risk at 8 bits too: per-position absmax lets one
  large channel coarsen the grid for all the others.
- Ours specifically: the out projection's input is the recurrence's read-out,
  a sum over resonant poles. Its dynamic range may be wider than a
  transformer's attention output. If a few channels ring loudly, the int8
  grid there is coarse. This is the open part.

## Hypothesis and prediction

Written 2026-10-23 12:10 UTC, before any run with `--act8`:

- **D** (wide ternary + int8 inputs, lr 4x, seed 0, 3K): **1.982–2.003 bpb**,
  a cost of 0.000–0.020 against C's mean 1.9825. Still ahead of the best
  float run (2.0039).

What would change our mind:
- A cost over 0.04 means the per-position absmax is too coarse for Vidya's
  read-out. The next arm would then keep the out projection's input in
  float and quantize only qkv's (layer-normed) input, to see which side
  hurts.
- A cost under 0.005 means int8 inputs are free here. The C port can then
  use int8 throughout the projections.

## Independent variable

The projections' inputs, float32 → int8 per position (`--act8`): x is scaled
by 127 over max |x| across the position's channels, rounded, clipped to
[−128, 127] and scaled back. The gradient passes straight through. It
applies to the input of every block's qkv projection (the layer-normed `u`)
and out projection (the recurrence read-out).

Unchanged: the recurrence, the poles, the norms, the head, the embedding
table (all float).

Held fixed, as ternary-projections' C: polar, dense, 4 layers, 64
harmonics, `--ternary`, `--lr-scale 4`, batch 16 x 2048, 3000 updates,
warmup 200 and cosine to 10%, seed 0, the same shard order, the same
scoring.

## Code

`harness/field_poles.py train --variant polar --mixing dense --layers 4
--harmonics 64 --batch 16 --updates 3000 --seed 0 --shards 2 --ternary --act8
--lr-scale 4`. `--act8` sets `zara_poles.ACT8`: `zara_poles.act8` is applied
through `projection_input` at the four projection call sites (chunked forward
and Sampler). `result.json` records `act8`, and `core_server.py` rebuilds the
same forward from it.

`harness/ternary_check.py OLD_DIR`, run on the lab CPU against the live
`~/field` harness (2026-10-23 ~12:05 UTC):
1. Both flags off: loss and all 31 gradients equal the old harness bit for
   bit.
2. `--ternary` on, `--act8` off: equal to the old ternary forward bit for
   bit.
3. `--act8` on: every input lies on its position's 255-level grid; the loss
   moves (5.685966 vs 5.685822); gradients reach every parameter.
4. The Sampler equals the chunked forward in float64 (max |diff| 1.8e-15).

In float32 the two paths differ by about 1e-6. That flips an occasional int8
rounding by a whole grid step: 18 of 64 positions differ by more than 1e-3,
worst 9.8e-3. This is a property of rounding, not a bug. A deployed integer
model should define its forward by one path, and the C port should be checked
against the Sampler.

A 4-update CPU smoke of `train --ternary --act8 --lr-scale 4` ran end to end
and wrote `ternary`, `act8` and `lrScale` into `result.json`.

## Hardware and cost

One community RTX 3090 at $0.22/h, about 75 min with the fetch and the full
held-out score: about $0.30.

## Data

As ternary-projections: OpenWebText shards 0–1 as bytes, 98,304,000 bytes
seen. Held out: the GPT-2 baseline's split (sha256 d0684974…, 18,538,212
scored bytes).

## Results

(none yet)

## Decided vs measured

Checked, not proved: the act8 forward puts every projection input on its
position's grid, and the two forward paths compute the same function (float64).
Measured: every bpb number.

## Verdict

(none yet)

## What it changes

If the cost is small, the projections in the C port are int8 × ternary
integer adds with one float multiply per output. The remaining float work is
the recurrence (4% of multiply-adds at 4L/64H), the norms and the head. The
recurrence is the target of the shift-decay and CORDIC experiments.
