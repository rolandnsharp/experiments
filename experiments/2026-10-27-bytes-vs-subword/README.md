# Bytes vs subword tokens for Vidya at equal training compute

**EXP-0033** (experiments.blah.dev), pre-registered 2026-10-26 14:53 UTC before any run.
**Status.** running: 3 pods on Roland's RunPod account (MARKS tok-b, tok-c2k, tok-c8k).

## Question

On the uConsole, decode speed is set by the bytes of weights read per step (EXP-0031), and a byte
model takes one step per byte. A subword model takes one step per token, about 3-4 bytes. At equal
training compute (same updates, same positions), does a small byte-level BPE vocabulary give Vidya
lower held-out bits per byte, and how much cheaper is decoding per byte of text?

## Prior art

- [Tao et al. 2024](https://arxiv.org/abs/2407.13623): the best vocabulary grows with compute; small
  models want small ones, since the embedding and the head grow with V.
- [Maliyawala & Lisal 2026](https://doi.org/10.5281/zenodo.23484616): the closest test to ours. At
  350M, same data, ~equal FLOPs, an 8K byte-level BPE scored 0.9155 vs 0.9634 bpb for a byte-patch
  model. One seed, a transformer.
- [H-Net](https://doi.org/10.48550/arxiv.2507.07955), [BLT](https://arxiv.org/abs/2412.09871),
  [MambaByte](https://arxiv.org/abs/2401.13660): byte models match or beat tokens, but only with a
  hierarchy (chunking, patches) or speculative decoding. A flat byte model like Vidya is the open case.
- The full review: [notes/2026-10-27-edge-architecture-literature.md](../../notes/2026-10-27-edge-architecture-literature.md).

## Arms

All: the factorial's best block (EXP-0029): polar, 4 layers, 20 harmonics + SwiGLU 600, `--select`,
`--conv 4`, `--gate 16`; lr 1x, 3,000 updates, batch 16 x 2048 positions, shards 0-1, seeds 1 and 2.

| arm | input | head | params | role |
|---|---|---|---|---|
| A | bytes, wire table | 256 | ~3.50 M | fxbest-1/2 (EXP-0029's confirmation runs, reused) |
| **B** | bytes, learned embedding (`--embed`) | 256 | 3,566,952 | **control** |
| C2k | BPE 2,048 (`--tokens owt-bpe2k`) | 2,048 | 4,595,560 | subword |
| C8k | BPE 8,192 (`--tokens owt-bpe8k`) | 8,192 | 8,122,216 | subword |

The learned table starts as random-phase unit phasors per group (the wire table's form), is
renormalized per group in the forward, and trains with Adam.

## Tokenizer

`harness/prepare_owt_tokens.py`: BPE over bytes (GPT-2's printable byte mapping, so every byte
sequence round-trips), pre-split before words, numbers, symbol runs and whitespace runs; id 0 the start
token (U+2402); trained on the first 100 MB of training shard 0 only. Measured before any run:

| | 2K | 8K |
|---|---|---|
| bytes per token, train / held-out | 2.927 / 2.932 | 3.791 / 3.795 |
| text read per run (98.3 M tokens) | ~288 MB | ~373 MB |

Byte arms read 98.3 MB.

## Scoring

The same 18,538,212 held-out bytes for every arm. Byte arms score window k's bytes k*2048+1 ..
k*2048+2048 given byte k*2048. Token arms tokenize exactly those bytes, prepend the start token and
score every token; bpb = bits / bytes. The byte model has one byte of context per window the token
model lacks (a bias against tokens, under ~0.003 bpb).

Decode multiply-adds per byte (computed, not predicted): bytes 1.52 M; C2k 0.70 M; C8k 1.02 M.

## Predictions (pre-registered)

1. B - A within ±0.03 bpb. Refuted if |B - A| > 0.05 (then the wire table matters).
2. C2k - B between -0.15 and -0.03. Refuted if C2k is not at least 0.02 below B.
3. C8k - C2k within ±0.04. Refuted if they differ by more than 0.04.

## Checks before any pod

On the lab CPU (`harness/check_tokens.py` against the live `~/field`): flags off, the loss and all 49
gradients bit-identical on the factorial's best block, and again at fxbest's exact size (95 tensors);
`--embed` adds only the table; a 2,048 vocabulary's Sampler equals the chunked forward (float64,
3.9e-15); `score_tokens` equals a hand sum by the Sampler. 2-update CPU smokes of B, C2k, C8k ran.

## Results

Pending.
