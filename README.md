# experiments

One folder per experiment on the oscillator language models (Vidya and its
predecessors). An experiment is a run whose answer we did not know: one
independent variable, everything else held fixed, a prediction written down
before the run, and a paper afterwards that says what happened and what it
changes.

## Rules

1. **Search first.** Before spending compute, look for the answer in the
   literature. The paper cites what was found (two or three sources, one line
   each on what they settled and why ours was still open). A search that
   settles the question gets a short note here instead of a run.
2. **Predict first.** The hypothesis and a numeric prediction go in the paper
   before the first update. The verdict is written against that prediction.
3. **One variable.** Say what moved and what was held fixed. When more than one
   thing moved, say so in the paper; it is still worth recording, but it is
   not a clean result.
4. **Seeds.** A single-seed gap is a hint. Two seeds is a result. Say which.
5. **The eval bundle never enters training.** CORE numbers come from
   evals.blah.dev scoring our endpoint; the held-out OpenWebText set is the
   GPT-2 baseline's split, reproduced byte for byte.
6. **Report compute beside accuracy.** Multiply-adds per byte, training FLOP,
   wall-clock, hardware and cost, so a result is comparable across model sizes.

## Layout

```
INDEX.md                      one row per experiment: date, variable, result, verdict
template/README.md            the paper's sections
experiments/<date>-<slug>/    README.md (the paper) + the run artifacts
harness/                      the Python harness as used (copies, with a note of the date)
```

Run artifacts kept per run: `result.json` (final held-out bpb, settings,
samples, learned poles), `compute.json` (MAC/byte, training FLOP),
`checkpoint.json` (the held-out curve on 256 windows), `train.log`,
`samples.txt`, and the CORE attempt records. Checkpoints (`params.pt`) stay
on the lab; the paper gives their path and md5.

## Where things run

- **lab**: an OVH VPS (8 vCPU, 22 GB), holds the data (`~/field/owt`), the
  runs (`~/field/runs*`), and serves the CORE endpoint at
  https://core.roland.blah.dev.
- **pods**: RunPod community RTX 3070 (Ajax's rule), 3090 when no 3070 is in
  stock; one pod at a time; the pod drives itself (`harness/pod-self-driver.sh`)
  and pushes results to the lab.
- **CORE**: evals.blah.dev, model `Vidya` (id in the attempt records), tiers
  smoke (50 items a task) and standard (500).
