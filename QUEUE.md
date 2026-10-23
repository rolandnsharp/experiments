# Queue

Experiments planned but not run, each with its prediction written now. One
variable each, on the 4L/32H polar dense harness at 3,000 updates (98 MB,
~40 min on a 3090, ~$0.15) unless the question needs more; two seeds before
a verdict. Taken from the parameter-golf leaderboard survey of 2026-10-22
(see the "prior art" lines); the rest of that board (quantisation,
test-time training, sliding evaluation, tokenizers) does not apply to a
byte model on CORE.

| # | experiment | variable | prediction | cost |
|---|---|---|---|---|
| 0 | real-decay-30k-seed-1 | the text-pole-form control at 30K, second seed (the first seed's rerun is already chained) | real 30K lands 0.03–0.05 bpb above polar's 1.8469 at seed 0 and the same at seed 1; if the gap is under 0.02 the pole claim does not hold at a full pass | 2 × 7 h |
| 1 | smear-gate | a learned per-channel gate mixing the previous byte's embedding into the current one before block 0 (~448 parameters, init 0.95) | −0.01 to −0.03 bpb at 3K; the gain shrinks at 30K | 2 × 40 min |
| 2 | depth-recurrence | blocks 1 and 2 run twice (6 block passes, 4 blocks' parameters) | −0.02 to −0.05 bpb at 3K, i.e. Vidya is compute-bound at this size; if under −0.01, parameter-bound | 2 × 55 min |
| 3 | muon-momentum-warmup-and-decay | Muon momentum 0.92 → 0.99 over the first 10% of updates and weight decay 0.04 (Diakosmos: fixed momentum, decay 0.01) | −0.01 to −0.02 bpb at 3K; no change in rank order between real and polar | 2 × 40 min |

## Notes on each

**1. smear-gate.** Prior art: the parameter-golf SmearGate records
(2026-03-19 onward) report it as a consistent small win for a BPE
transformer, where it supplies bigram context the attention would otherwise
have to learn. For a byte model it is cheaper still, but the poles already
resonate on byte pairs, so the gain may be smaller. Held fixed: everything
else in the 4L/32H harness.

**2. depth-recurrence.** Prior art: depth recurrence on layers 4 and 5 in
several 16 MB records, and the organisers' request for universal-transformer
entries. The reason to run it here is the compute story: a looped block adds
multiply-adds but no parameters, so the result says which of the two Vidya
is short of at 3.4 M parameters. Report MAC/byte beside bpb.

**3. muon-momentum-warmup-and-decay.** Prior art: the Muon weight-decay and
momentum-warmup records; we inherit Diakosmos's optimiser constants untouched
and have never varied them. Two flags in `diakosmos_oracle.Update`.

## Parked: speech recogniser

- **stt-punctuation** (after accuracy work; Roland 2026-10-23: "since it's bolt-on we don't need
  to rush"). Punctuation and capitals as a separate text model on each finished phrase, since
  LibriSpeech transcripts carry neither. Baseline: sherpa-onnx's online punctuation model. Then
  our own: a small byte-level oscillator model (Vidya) trained to restore punctuation on modern
  text plus Roland's writing, possibly with the recogniser's pause lengths as a hint. Prediction
  to be written when it is scheduled.

## Not queued, and why

- Test-time training and sliding-window scoring: they train or condition on
  the scored stream; CORE scores a fixed endpoint.
- Int6/GPTQ quantisation and artifact compression: no size budget on CORE.
- Bigram hash embeddings: overlaps with what the poles do; revisit if
  smear-gate shows a bigram gain worth chasing further.
- SentencePiece 8192 and CaseOps: we are byte level by design.
- Orthogonal projection init: predicted null (within ±0.01 bpb); dropped
  2026-10-22 rather than spend two seeds confirming it. Revisit only if a
  run shows early-training instability in the dense projections.

## The divergence queue (added 2026-10-22)

The headline result is the pole itself at equal compute (text-pole-form).
These three ask whether the oscillator bank can carry more of the model, so
that Vidya differs from Diakosmos in structure and not only in the channel.
Each on the 4L/32H polar dense harness at 3K, two seeds, the prediction
written now.

| # | experiment | variable | prediction | cost |
|---|---|---|---|---|
| 4 | frequency-resolved-readout | the out projection replaced by per-pole energy and phase features (|S|^2, cos and sin of the phase per channel, 3 x 56 words a head) into a small [168 x 64] projection a head; parameters fall from 3.39 M to ~2.6 M, MAC/byte by ~20% | +0.03 to +0.08 bpb at equal updates (a loss), but ahead of the dense model per MAC/byte; if the loss is under 0.02 the dense out projection was carrying little | 2 x 40 min |
| 5 | pole-coupling | a learned [8 x 8] complex mixing of a head's channel states each position (S' = C S + k v), C initialised to the identity; +7 x 128 parameters a block | -0.01 to -0.03 bpb at 3K; under -0.005 means the channels are already as expressive as the read-out can use | 2 x 45 min |
| 6 | channels-per-head | 7 heads x 16 channels at width 448 (the harmonic projections double, the dense projections unchanged; +0.47 M parameters, +4% MAC/byte) against the 8-channel baseline | -0.02 to -0.04 bpb at 3K; per parameter about the same as widening the dense projections, which would make it a null result for the oscillator claim | 2 x 45 min |

Order: 0 (the control, needed before anything is sent), then 4, then 5,
then 1–3 and 6. 4 is the one that would make the architecture different; 5
and 6 say whether the bank is the bottleneck or the projections are.
