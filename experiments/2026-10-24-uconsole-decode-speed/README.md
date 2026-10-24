# Ternary vs int8 vs int4 decode speed on the uConsole CPU

**EXP-0031** (experiments.blah.dev). **Date.** 2026-10-24. **Status.** done.
Not pre-registered: the runs came first. The prediction (ternary 3-4x faster
than int8 once decoding is RAM-bound) was stated in conversation with Roland
before them.

## Question

On the uConsole's RK3399 (2x Cortex-A72 at 1.8 GHz, 4x Cortex-A53 at 1.4 GHz,
ARMv8.0 without the dot-product instructions, 3.8 GB LPDDR4), how fast does
single-stream decoding run with ternary weights compared with int8 and int4,
and at what model size does memory bandwidth set the speed?

ternary-wide-30k (EXP-0013) concluded "ternary does not pay on CPUs". That
came from our C forward on the x86 lab, whose ternary path stores each weight
as an int8 byte and runs the same int8 multiply-adds. It timed the wide model's
3.8x more multiply-adds through one kernel, not a ternary kernel.

## Method

- Read bandwidth: a NEON loop summing a 256 MB buffer, best of 5, pinned with
  `taskset`.
- Decode: llama.cpp 3642238 (`GGML_NATIVE`, `-march=armv8-a`), `llama-bench
  -p 0 -n 64 -r 3`, pinned to the A72s (2 threads), the A53s (4) or all six.
- Models: SmolLM2-135M/360M/1.7B-Instruct GGUF (bartowski) at Q8_0 and Q4_0.
  TQ2_0 (ternary, 2.06 bits a weight) made with `llama-quantize
  --allow-requantize` from the Q8_0 file. Its weights are not a trained
  ternary model and its text is nonsense; speed depends only on tensor types
  and shapes.
- TQ2_0 needs rows that are multiples of 256. 135M (width 576) and 360M (960)
  fell back to Q4_0 for most tensors, so only 1.7B (2048) is tested ternary.
- Plugged in, all six cores online.

## Results

Best decode tokens/s over the three core sets:

| model | Q8_0 | Q4_0 | TQ2_0 |
|---|---|---|---|
| 135M | 34.4 | 41.2 | n/a |
| 360M | 14.5 | 15.4 | n/a |
| 1.7B | 3.31 | 4.71 | **7.89** |

By core set (A72x2 / A53x4 / all six), 1.7B: Q8_0 3.31 / 2.39 / 3.09;
Q4_0 3.07 / 2.69 / 4.71; TQ2_0 5.23 / 4.62 / 7.89.

RAM read: 6.44 GB/s on one A72, 6.63 on both, 1.88 on one A53, 6.19 on four,
6.37 on all six.

Bytes streamed per second at 1.7B: Q8_0 6.0 GB/s (at the RAM ceiling), Q4_0
4.7, TQ2_0 3.9 (compute-bound).

## Verdict

Prediction missed: ternary is 2.4x int8 (and 1.7x int4), not 3-4x, because
the TQ2_0 kernel runs out of compute before it runs out of RAM bandwidth. A
faster ternary kernel (lookup tables, as T-MAC and bitnet.cpp's TL kernels)
has up to ~1.6x of headroom before the RAM wall.

What it changes: EXP-0013's "ternary does not pay on CPUs" holds only for its
cache-sized 3-4 MB model. On this chip, for models larger than the caches,
ternary is the fastest format. Int8 is RAM-bound from ~360M up, and the two
A72s alone saturate RAM, so the A53s add nothing to int8; the lower-bit formats
gain from all six cores. A ~1-2B ternary model decodes at ~8 tok/s here
(about 6 words/s), which is interactive.

## Limitations

Speed only. One run of 3 reps per cell. One llama.cpp commit; TQ1_0 and
lookup-table kernels not tested. Prompt processing not measured.
