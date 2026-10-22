# experiments

Experiments on AI models, written up so that others can check them, build on
them, and not repeat them. An experiment is a run whose answer we did not know:
one independent variable, everything else held fixed, a prediction written down
before the run, and a paper afterwards that says what happened and what it
changes. Failed runs, null results and mistakes are recorded with the same care
as wins; they are the part that saves the next person the most.

The collection so far covers oscillator language models (Vidya and its
predecessors) and an oscillator speech recogniser (folders `*-stt-*`, harness
in `harness/stt/`), but nothing here is specific to them: any model, any
modality, any lab member.

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
5. **Evaluation data never enters training.** Name the held-out sets and how
   they were kept out (for example: CORE numbers come from evals.blah.dev
   scoring our endpoint; the held-out OpenWebText set is the GPT-2 baseline's
   split, reproduced byte for byte; LibriSpeech test sets are never trained on).
   If anything was tuned on the reported set, say so.
6. **Report compute beside accuracy.** Multiply-adds per unit of input (byte,
   token, second of audio), training FLOP, wall-clock, hardware and cost, so a
   result is comparable across model sizes.
7. **Record everything that ran.** Crashes, out-of-memory deaths, broken hosts
   and abandoned arms go in the paper with their cost. A rule broken (no
   prediction written, two variables moved) is stated, not hidden.
8. **Rerunnable.** A paper carries enough (code, data, settings, seeds) for a
   reader who doubts the result to rerun it. Review is welcome, not required;
   an objection, when someone writes one, goes into the paper.

## Experiments and proofs

Our models run in pure arithmetic, so it is tempting to ask whether a run is
a proof. It is not, and the repo keeps the two apart. (The examples below are
from the Vidya work in Alpha; the distinction holds for any model.)

**An experiment measures.** A training run's result is a property of one
dataset, one optimizer trajectory and one seed. "Polar beats real by 0.04
bpb" is a measurement with variance: the seed runs in this repo show the
spread that comes with it, and a different shard order or a different seed
could move it. No proof turns a measurement into a theorem, because it is not
one. Every bpb and CORE number here is a measurement, and the paper says so.

**A proof decides.** Some claims hold for all inputs and are decided by the
Alpha checker, not measured: that a kernel computes the text bit for bit
(the SM86 model checks), that the parameter count and the multiply-adds per
byte are what the layout says, that grouped mixing equals the block-diagonal
restriction of the dense product, that the adjoint is the derivative of the
forward text. These live as laws in the Vidya system (`systems/vidya/tests/
qualification/Proof/`), in Alpha because Alpha is the language whose checker
decides them. A paper cites them as decided and links the root that holds them.

**What a proof is for here.** It tells you that what ran is what the text
says, so a measurement is of the architecture and not of a bug, and that two
things claimed equal (two layouts, a kernel and its text) are equal. It never
tells you the model learns, or that one form of the model learns better; only
a run does that, and only within its seeds and data.

**What goes where.** A claim about the architecture that holds for all inputs
(stability of the pole under its parameterisation, the equal-compute
equivalence of two layouts) belongs as an Alpha law in the Vidya system,
linked from the paper. A claim about a trained model belongs in the paper's
results, with its seeds. Each paper's "decided vs measured" section says which
of its claims is which, and nothing is promoted from the second kind to the
first by this repo.

## Layout

```
INDEX.md                      one row per experiment: date, variable, result, verdict
QUEUE.md                      experiments planned, with their predictions written before any run
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
