# Task 1 — Instrument the current engine

**Prereq:** `general.md` (esp. "Why the current engine is slow").
**Goal:** replace hand-waving with real numbers about *where* time goes today, so the
later speedups can be attributed. Two separate exercises; each is its own session.

## 1a. Empirical profiling (a measurement session)

Using the 784→32→10 net and the training/eval loop from `notebooks/mnist.ipynb`:

- Count objects: how many `Value` nodes are in the graph? How many `op.forward` /
  `op.backward` calls happen per forward pass and per backward pass? (Instrument, or
  reason from the graph structure and confirm.)
- Profile one forward pass and one forward+backward with `cProfile` (and/or a manual
  timer). Where does wall-clock actually go — the graph traversal in `Value.forward`,
  the `op.forward` dispatch, the `list`/`zip` comprehensions, the SGD update loop?
- Confirm the anchor: is a forward pass really ~30 ms? Split forward vs backward.
- Sanity: what fraction of time is *actual float arithmetic* vs Python overhead?

**Deliverable:** `1_interpreter_tax_writeup.md` — the counts, a profile breakdown, and
a one-paragraph "the slowness is X% overhead" conclusion.

## 1b. Theoretical FLOP/s from first principles (a separate session)

This one assumes **no prior comfort** with FLOP counting or the roofline model — the
point is to build the reasoning from scratch. Work through, in order:

- What counts as a FLOP? Why is a multiply-accumulate usually counted as 2.
- How many FLOPs are in one forward pass of the 784→32→10 net? (Count the MACs in
  each matmul; decide how to count tanh/softmax.) Then one backward pass.
- Divide by the measured wall-clock (the ~30 ms anchor, or the real number from 1a)
  to get **achieved FLOP/s**.
- Derive the CPU's **theoretical peak** FLOP/s for this chip: cores × clock ×
  (AVX2 lanes) × (2 for FMA) × (2 for mul+add). Then: what % of peak are we hitting?
- Interpret: the gap between achieved and peak *is* the combined interpreter +
  representation tax. This number reframes the whole project.

**Deliverable:** `1_flops_writeup.md` — the derivation, both numbers, the ratio, and
what it implies about the ceiling the later phases are chasing.

## Session kickoff prompt
> "Read `data/omer_questions/simd_speedup_plan/general.md` and
> `1_instrument_current_engine.md`. Let's do exercise **1a** [or **1b**]. [For 1a:]
> Help me profile the current scalar engine on one MNIST example — object/op counts
> and a cProfile breakdown of forward vs backward — and write `1_interpreter_tax_
> writeup.md`. [For 1b:] I have no background in FLOP counting; build the reasoning
> from scratch to get achieved FLOP/s and % of my CPU's theoretical peak, and write
> `1_flops_writeup.md`."
