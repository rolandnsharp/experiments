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
