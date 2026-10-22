# <title: the variable and the model>

**Date.** YYYY-MM-DD. **Status.** planned | running | done.

## Question

One sentence: what we did not know.

## Prior art

- <source> — what it settled; why our case was still open.
- <source> — ...

## Hypothesis and prediction

What we expect and a number (bpb, CORE, or a gap), written before the run.
What result would change our mind.

## Independent variable

What moved. What was held fixed (model size, data, updates, batch, seed,
optimizer, schedule). If more than one thing moved, say so.

## Code

Harness and commit or file hash. The diff that matters, quoted or linked.

## Hardware and cost

GPU, cloud, price, wall-clock, dollars.

## Data

Training bytes (shards, bytes seen, passes), held-out set and its hash, what
was excluded.

## Results

Held-out bpb (all scored bytes) per arm; the curve on the 256-window subset;
CORE if scored; compute (MAC/byte, training FLOP). Samples if they say
something.

## Verdict

Against the prediction. Seeds. Caveats.

## What it changes

The next experiment this result argues for, or against.
