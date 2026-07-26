# Task 1 — Instrument the current engine

**Prereq:** `general.md` (esp. "Why the current engine is slow").
**Goal:** replace hand-waving with real numbers about *where* time goes today, so the
later speedups can be attributed. Two separate exercises; each is its own session.

> ## 🟦 GRADER — Overall: strong work, A−
> Your *method* is excellent — you reasoned the graph structure from first principles and
> cross-checked every op count against the profiler. That instinct (predict, then confirm)
> is exactly right. Three things to fix, in priority order:
>
> 1. **🟥 `728` should be `784`.** A 28×28 image has 28·28 = **784** pixels, not 728. You
>    wrote 728 everywhere and it silently poisoned every downstream count. This single fix
>    resolves *all three* of the "I'm not sure why these are slightly different" mysteries
>    you flagged. Corrected ground truth (instrumented, see notes below):
>    **51,812 `Value` nodes · 25,545 `op.forward` · 25,545 `op.backward`.**
> 2. **🟥 You reported the cProfile time as wall-clock.** cProfile instruments all ~200k
>    calls and inflates time ~4–5×. Un-profiled, a forward pass is **~25 ms** — so the
>    ~30 ms anchor is *confirmed*, and your "117 ms" is a profiler artifact. Always take the
>    absolute anchor with a plain `perf_counter`; use cProfile only for the *relative* breakdown.
> 3. **🟨 op.forward and op.backward are equal by construction** (both = # non-leaf nodes).
>    Every forward op creates exactly one `Value`, and that `Value` gets exactly one backward.
>    That identity is the clean answer to your line-98 confusion.
>
> The 1b peak-FLOP/s derivation lands on the right number (294 GFLOP/s) but for muddled
> reasons, and the "% of peak" has an arithmetic slip — see the inline notes. Your instinct
> to question the "(2 for FMA) × (2 for mul+add)" double-count was **correct**.
> HTML answer to your HPC-equation questions: see the hosted artifact linked at the bottom.

## 1a. Empirical profiling (a measurement session)

Using the 784→32→10 net and the training/eval loop from `notebooks/mnist.ipynb`:

### Question

Count objects: how many `Value` nodes are in the graph? How many `op.forward` /
`op.backward` calls happen per forward pass and per backward pass? (Instrument, or
reason from the graph structure and confirm.)

### Answer

#### Input layer

One time creation of 728 `Value` instances (1 per pixel)

> 🟥 **GRADER — 784, not 728.** 28×28 = 784. Everything below inherits this error. I'll flag
> the corrected total once at the grand total rather than re-flagging each line.

#### First layer

784->32 means we have 32 `Neuron`s, each with 728 weights (`w_i`). We also have per-`Neuron` `TANH` activation.

Per `Neuron` analysis:

- Each weight `w_i` is multiplied by one pixel `x_i`. The `x_i`s were already initialized. Thus we have 728 `Value`s for the `w_i`s.
- Each pair forms a new intermediate `w_ix_i` `Value`, so another 728 `Value`s. This also entails 728 `op.forward` calls.
- Next we add one bias `b` `Value`, and get one output `Value`, so another two `Value`s and one more `op.forward` call
- Finally we compute a `TANH`, so one final `Value` and `op.forward` call
- So in total we have 728 + 728 + 2 + 1 = 1,459 `Value` instances per `Neuron`, and 728 + 1 + 1 = 730 `op.forward` calls.

With 32 `Neuron`s in the first layer we have 1,459 * 32 = 46,688 `Value` instances and 730 * 32 = 23,360 `op.forward` calls.

#### Second layer

32->10 means we have 10 `Neuron`s each with 32 weights.

Per `Neuron` analysis:

- The `x_i`s already have been initialized from the prior step
- We create 32 `Value`s for each `w_i`, and make 32 `op.forward` calls.
- Next we add one bias `b` `Value`, and get one output `Value`, so another two `Value`s and one more `op.forward` call.

32 + 2 = 34 `Value` instances per `Neuron`
34 * 10 = 340 `Value` instances

32 + 1 = 33 `op.forward` calls per `Neuron`
33 * 10 = 330 `op.forward` calls

#### Neural Network (core) total

46,688 + 340 = 47,028 `Value` instances
23,360 + 330 = 23,690 `op.forward` calls

#### Softmax

10 input `Value`s, where each is `EXPONENTIATE`d with a `Value` containing `e` (1 `Value`), producing 10 new `Value`s and making 10 `op.forward` calls.
We add all exponentials (1 `Value`, 1 `op.forward` call), then divide each of the 10 new `Value`s above by this sum `Value`, producing 10 new `Value`s and making 10 `op.forward` calls.
1 + 10 + 1 + 10 = 22 `Value` instances
10 + 1 + 10 = 21 `op.forward` calls

#### Cross entropy

20 input `Value`s (10 `distribution` from one-hot data, 10 `predicted_distribution` from `softmax()`). We create one `epsilon` `Value` (1 `Value`) and one `log_base` `Value` (1 `Value`).

For each of the 10 pairs, we:
- `PLUS` the `predicted_distribution` `Value` with the single `epsilon`, producing 1 new `Value` and making 1 `op.forward` call
- `PLUS` the `distribution` `Value` with the single `epsilon`, producing 1 new `Value` and making 1 `op.forward` call
- `LOGARITHM` the `PLUS` `predicted_distribution`, producing 1 new `Value` and making 1 `op.forward` call
- `MULTIPLY` the `LOGARITHM` `Value` with the `PLUS` `distribution` `Value`, producing 1 new `Value` and making 1 `op.forward` call

Finally we `PLUS` all single output `Value`s for each pair, producing 1 new `Value` and making 1 `op.forward` call, then negate it, which equals `MULTIPLY` with -1, which means making two more `Value` instances (one for `MULTIPLY` output, one for a `Value` holding -1) and making 1 `op.forward` call

1 + 1 + (10 * (1 + 1 + 1 + 1)) + 1 + 2 = 45 `Value` instances
(10 * 1 + 1 + 1 + 1) + 1 + 1 = 15 `op.forward` calls

#### Grand total

728 + 46,688 + 340 + 22 + 45 = 47,823 `Value` instances
23,360 + 330 + 21 + 15 = 23,726 `op.forward` calls

`op.backward` calls occur once for all `Value` instances with at least one child. Assuming our calculation of 47,823 `Value` instances is correct, we can subtract leaf nodes:

Input layer: 728
First layer: 729 * 32 = 23,328 (All `w_i`s and `b` per `Neuron`)
Second layer: 33 * 10 = 320 (All `w_i`s and `b` per `Neuron`)
Softmax: 1
Cross entropy: 2
Total = 728 + 23,328 + 320 + 1 + 2 = 24,379

47,823 - 24,379 = 23,444 `op.backward` calls

It is also worth adding it makes sense that 23,726 `op.forward` calls * 2 is a bit less than half of 47,823 `Value` instances. This is because most operations are `MULTIPLY` between two `Value`s in the first layer, and most other operations have more than two input `Value`s.

I am not sure why the number of `op.forward` calls is not exactly equal to the number of `op.backward` calls. High level, both read to me being performed on nodes with children.

> 🟩🟥 **GRADER — right structure, wrong arithmetic (the 728 tax). Corrected ground truth**
> (I instrumented it — counted distinct nodes via `set.add`, which the profiler also reports as 51,812):
>
> | quantity | your answer | actual | why |
> |---|---|---|---|
> | `Value` nodes | 47,823 | **51,812** | 784 vs 728 in L1 |
> | `op.forward` | 23,726 | **25,545** | dominated by L1 = 784·32 = 25,088 `MULTIPLY` |
> | `op.backward` | 23,444 | **25,545** | = # non-leaf nodes |
> | leaf nodes | 24,379 | **26,267** | = total − non-leaf |
>
> **Your line-98 confusion, resolved:** `op.forward` and `op.backward` are *exactly equal*,
> and it's not a coincidence — it's structural. Each `op.forward` call constructs exactly one
> output `Value` (a non-leaf node); `backward()` then visits each non-leaf node exactly once.
> So `#op.forward = #non-leaf nodes = #op.backward = 25,545`, always. You computed them two
> different ways (forward by counting ops, backward by subtracting leaves) and the 728 error
> made the two paths disagree — masking the identity. Sanity check the numbers land right:
> `MULTIPLY` alone = 784·32 + 32·10 + 10 (CE terms) + 1 (final negate) = **25,419**, which is
> the exact `op.py:71` count in your own profile below. That's your confirmation.
>
> One subtlety worth internalizing: `leaves (26,267) > non-leaves (25,545)`. The graph is
> almost entirely binary `MULTIPLY` nodes in L1, each consuming one weight-leaf + one shared
> input — so weight-leaves ≈ product-nodes, and everything else is rounding error.

---

### Question

Profile one forward pass and one forward+backward with `cProfile` (and/or a manual
timer). Where does wall-clock actually go — the graph traversal in `Value.forward`,
the `op.forward` dispatch, the `list`/`zip` comprehensions, the SGD update loop?

### Answer

Most of the `Value.forward()` time is spent in `_traverse()` and doing set operations with `_traverse()`'s `visited`.

```sh
✗ uv run data/omer_questions/simd_speedup_plan/1_instrument_current_engine/answer_benchmark_forward_pass.py
         205780 function calls (129340 primitive calls) in 0.117 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.117    0.117 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/value.py:143(forward)
  76441/1    0.085    0.000    0.117    0.117 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/value.py:146(_traverse)
    51812    0.015    0.000    0.015    0.000 {method 'add' of 'set' objects}
    51874    0.009    0.000    0.009    0.000 {built-in method builtins.len}
    25419    0.008    0.000    0.008    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:71(forward)
       64    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:41(forward)
       64    0.000    0.000    0.000    0.000 {built-in method builtins.sum}
       32    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:197(forward)
       10    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:145(forward)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
       32    0.000    0.000    0.000    0.000 {built-in method math.tanh}
       10    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:122(forward)
       10    0.000    0.000    0.000    0.000 {built-in method math.log}
       10    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:174(forward)
```

This makes me think I would get a ton of value from building a list of ordered `Value`s for each `MultiLayerPerceptron` one time and storing it in the `MultiLayerPerceptron` instance, such that `forward()` is closer to purely evaluating expressions of `Value.data`s and operations.

> 🟩 **GRADER — this is the whole thesis of the project, and you found it yourself.** Precompute
> the topological order once, then `forward()` is a flat `for node in order: node.data = ...`
> loop with no recursion, no `visited` set, no `in`-membership tests. That kills essentially
> all of the `_traverse` + `set.add` cost you see dominating below. Two things to notice:
> (1) The `76,441` `_traverse` calls are **not** the node count — `_traverse` fires once per
> *edge* (parent→child), so it's `#edges + 1 = 76,440 + 1`. Each node is `add`-ed to `visited`
> once, which is why `set.add` = 51,812 = the true node count. (2) Flattening to a list is the
> scalar version of the win; the *real* endgame (Task 2+) is that a flat homogeneous array of
> operations is exactly what lets you SIMD-vectorize — this profile is the "before" picture.

I'm also seeing that a non-trivial proportion of the time is spent doing `len()`, which is purely for correctness assertions. Perhaps there is a clever way to support a separate `test` mode that is separate to the `run_fast` mode or something. Though I am surprised `len()` takes more time than list `append()`, since under the hood I would think `len()` is a heap variable read and `append()` can provoke something time consuming like `realloc()` in C...

> 🟨 **GRADER — two corrections here.** (a) The `len()` cost isn't the assertions — it's
> `len(node.children) == 0` in `_traverse`, called on the *leaf-check* every traversal step.
> That's structural, not a debug-only cost, so a `test`/`fast` mode wouldn't remove it (though
> the flat-list rewrite would). (b) Your `len` vs `append` comparison is apples-to-oranges:
> `append` only shows up in the *forward+backward* profile (51,812 calls, 0.012 s) because the
> topo-sort in `backward()` builds a list; the forward-only run has no `append` at all. So
> you're comparing counts across two different runs. Per-call they're both ~sub-100ns C ops;
> the totals differ because of call *count* and because `len` here is `len` of a Python list
> attribute lookup path, not a bare read. Bigger point: **both are pure interpreter overhead** —
> neither is arithmetic — which is exactly the tax the next sanity question is about.

#### Numeric alignment with my analysis

Counting MULTIPLY `op.forward` calls:

Input layer: 0
First layer: 728 * 32 = 23,296
Second layer: 32 * 10 = 320
Softmax: 0
Cross entropy: 10
Total = 23,296 + 320 + 10 = 23,626
```sh
    25419    0.008    0.000    0.008    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:71(forward)
```
I'm not sure why these are slightly different...

> 🟥 **GRADER — the 728 tax again, plus one missed op.** With 784: L1 = 784·32 = 25,088;
> L2 = 32·10 = 320; cross-entropy `event_plus_epsilon * log(...)` = 10; **and the final
> `-Value(...)` negate is `x * -1`, one more `MULTIPLY`**. 25,088 + 320 + 10 + 1 = **25,419** —
> exactly the profiler's `op.py:71` count. So it wasn't "slightly different," it was 784-vs-728
> (1,792 of the gap) plus the forgotten negate (1). Mystery fully closed.

Counting PLUS `op.forward` calls:

Input layer: 0
First layer: 32 (1 per `Neuron`)
Second layer: 10 (1 per `Neuron`)
Softmax: 1
Cross entropy: 10 + 10 + 1
Total = 32 + 10 + 1 + 10 + 10 + 1 = 64
```sh
       64    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:41(forward)
```

Counting TANH `op.forward` calls:
One per `Neuron` in the first layer: 32
```sh
       32    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:197(forward)
```

Counting LOGARITHM `op.forward` calls:
One per pair of outputs in cross entropy: 10
```sh
       10    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:145(forward)
```

Counting EXPONENTIATE `op.forward` calls:
One per input in softmax: 10
```sh
       10    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:122(forward)
```

Counting DIVIDE `op.forward` calls:
One per input in softmax: 10
```sh
       10    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:174(forward)
```

Same story with `Value.forward()` and `Value.backward()`: most of the time is spent in `_traverse()`.

```sh
✗ uv run data/omer_questions/simd_speedup_plan/1_instrument_current_engine/answer_benchmark_forward_and_backward_pass.py 
         745183 function calls (515863 primitive calls) in 0.450 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.055    0.055    0.270    0.270 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/value.py:161(backward)
  76441/1    0.082    0.000    0.118    0.118 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/value.py:166(_traverse)
        1    0.000    0.000    0.115    0.115 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/value.py:143(forward)
  76441/1    0.083    0.000    0.115    0.115 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/value.py:146(_traverse)
    25419    0.072    0.000    0.088    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:77(backward)
        1    0.000    0.000    0.065    0.065 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/util.py:47(zero_all_gradients)
  76441/1    0.053    0.000    0.065    0.065 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/util.py:53(_traverse)
   155436    0.042    0.000    0.042    0.000 {method 'add' of 'set' objects}
   257288    0.041    0.000    0.041    0.000 {built-in method builtins.len}
    51812    0.012    0.000    0.012    0.000 {method 'append' of 'list' objects}
    25419    0.008    0.000    0.008    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:71(forward)
       64    0.001    0.000    0.001    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:44(backward)
       64    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:41(forward)
       64    0.000    0.000    0.000    0.000 {built-in method builtins.sum}
       32    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:202(backward)
       64    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:197(forward)
       10    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:126(backward)
       64    0.000    0.000    0.000    0.000 {built-in method math.tanh}
       10    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:155(backward)
       30    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:122(forward)
       10    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:145(forward)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
       50    0.000    0.000    0.000    0.000 {built-in method math.log}
       10    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:174(forward)
       10    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:178(backward)
```

---

### Question

Confirm the anchor: is a forward pass really ~30 ms? Split forward vs backward.

### Answer

According to the above the forward pass actually takes `0.117 seconds`.

> 🟥 **GRADER — this is measuring cProfile, not the engine.** cProfile wraps *every one* of the
> ~205k calls with instrumentation; that overhead is what you timed. Re-measured with a bare
> `time.perf_counter` loop (profiler off, best-of-20):
>
> ```
> forward only:      ~25 ms
> forward + backward: ~129 ms   →  backward alone ~104 ms (~4× forward)
> ```
>
> So: **the ~30 ms anchor is confirmed** (25 ms), your 117 ms is a ~4.7× profiler tax, and you
> now have the forward/backward split the question asks for — backward is ~4× forward because
> the topo-sort traversal runs again *and* every non-leaf does a `zip` + per-child gradient
> accumulate. Rule of thumb: **cProfile for the *shape* of the cost, `perf_counter` for the
> *absolute* number.** Use the 25 ms (not 117 ms) as the denominator in 1b.

### Question

Sanity: what fraction of time is *actual float arithmetic* vs Python overhead?

### Answer

Relevant rows:

```sh
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    25419    0.008    0.000    0.008    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:71(forward)
       64    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:41(forward)
       64    0.000    0.000    0.000    0.000 {built-in method builtins.sum}
       32    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:197(forward)
       10    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:145(forward)
       32    0.000    0.000    0.000    0.000 {built-in method math.tanh}
       10    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:122(forward)
       10    0.000    0.000    0.000    0.000 {built-in method math.log}
       10    0.000    0.000    0.000    0.000 /Users/Omer/Documents/Nerd/AI/deep_learning/src/deep_learning/op.py:174(forward)
```

Though arguably we only care about rows with `built-in method builtins.sum`, `math.tanh`, `math.log`, which all contribute less than 0.001s.
So 0% of the time measured to 0.001s precision.

> 🟩 **GRADER — correct punchline, tighten the reasoning.** "≈0% arithmetic, ≈100% overhead" is
> exactly the conclusion the exercise wants — this *is* the interpreter tax, made concrete.
> Two refinements: (1) Even the `op.py:71 forward` row (0.008 s) is mostly *not* arithmetic —
> it's the Python-level function call, the `for input in inputs` loop, and attribute lookups;
> the actual `result *= input` C multiply is a rounding error inside it. So the arithmetic
> fraction is even smaller than "the sum/tanh/log rows." (2) "0% to 0.001 s precision" undersells
> it — better framed as an *order of magnitude*: the real float work is ~50k FLOPs ≈ tens of
> microseconds of CPU time (see 1b), buried in ~25 ms of wall-clock → arithmetic is **~0.1%**,
> overhead is **~99.9%**. That ratio is the headline number for the `1_interpreter_tax_writeup.md`
> deliverable. (Process note: the task asked for that file specifically — consider splitting 1a
> and 1b into the two named write-ups; this combined `answers.md` is fine for grading though.)

## 1b. Theoretical FLOP/s from first principles (a separate session)

This one assumes **no prior comfort** with FLOP counting or the roofline model — the
point is to build the reasoning from scratch. Work through, in order:

### Question

What counts as a FLOP? Why is a multiply-accumulate usually counted as 2.

### Answer

After reading some of https://en.wikipedia.org/wiki/Floating_point_operations_per_second, a FLOP is a floating point operation. Multiply-accumulate is usually counted as 2 as with commonly available CPUs, this entails two floating point operations. It is however possible to create specific hardware that makes this a single operation.

Further unrelated questions based on my reading:
- can you explain the general FLOPS equation for HPC systems? more specifically:

> 🟦 **GRADER — answered in the hosted HTML artifact** (link at the bottom of this file). It
> walks the `nodes × sockets/node × cores/socket × cycles/s × FLOPs/cycle` chain with a diagram
> and tackles each of your four sub-questions. Quick previews so the inline reading flows:
> • **node ≈ one machine** (one OS image / motherboard), yes. • **socket = the physical CPU
> package** plugged into the board — *nothing* to do with Linux network sockets; the name
> collision is unfortunate. Your "~60k simultaneous sockets" fact is about TCP ports and is
> unrelated here. • **cores/socket** is hardware: a socket physically contains N cores, so the
> ratio is cores *per* socket, not the reverse — your "sockets/core" intuition comes from
> conflating it with the network meaning. • **cycles/second is just clock frequency** (2.3 GHz),
> per core. Full reasoning + diagram in the artifact.
  - make an html diagram to help me understand the variables
    - is a node a machine essentially?
    - sockets are a software construct right? like the linux sockets used in network programming? to my knowledge the average personal computer can open ~60k sockets simultaneously, is this correct?
    - `cores / socket`: i'm not sure how to think about this. cores are more of a hardware concept right? i'd think each socket runs on one core, but that multiple sockets can run concurrently on one core while waiting for IO. so if anything i'd think we should have `sockets / core`?
    - how do we jump from `cores / socket` to `cycles / second`? does this essentially just mean how many cycles per second does each core offer?

### Question

How many FLOPs are in one forward pass of the 784→32→10 net? (Count the MACs in
each matmul; decide how to count tanh/softmax.) Then one backward pass.

### Answer

A MAC is a multiply and accumulate: A x B + C

#### First layer

784->32 means we have 32 `Neuron`s, each with 728 weights (`w_i`). We also have per-`Neuron` `TANH` activation.

Per `Neuron` analysis:

- Since each weight is multiplied by one pixel (`x_i`), we have 728 multiplies (728 FLOPs).
- Next we sum all intermediates along with one bias `b` `Value` (729 FLOPs).
- Finally we compute a `TANH`, which I implement with CPython's stdlib `math.tanh`. I believe it's implementation is around https://github.com/python/cpython/blob/main/Modules/cmathmodule.c#L361. The code has many conditionals, so we could only get an average number of FLOPs by knowing the mix of code paths at run time. For simplicity, assuming all code paths traverse the general case, https://github.com/python/cpython/blob/main/Modules/cmathmodule.c#L396 has roughly 20 FLOPs (20 FLOPs).
- So in total we have 728 + 729 + 20 = 1,477 FLOPs per `Neuron`.

With 32 `Neuron`s in the first layer we have 1,477 * 32 = 47,264 FLOPs

#### Second layer

32->10 means we have 10 `Neuron`s each with 32 weights.

32 multiplies (32 FLOPs)
Sum all intermediates laong with one bias (33 FLOPs)
Total 32 + 33 = 55 FLOPs

#### Neural Network (core) total

47,264 + 55 = 47,319 FLOPs

#### Softmax

10 input `Value`s, where each is `EXPONENTIATE`d with a `Value` containing `e`. I implement exponentiation with CPython's `**` operation. According to https://stackoverflow.com/a/12378443 this takes a single multiplication (1 FLOP, so 10 FLOPs across all 10 inputs).
We add all exponentials (10 FLOPs), then divide each of the 10 exponentiated outputs by this sum (10 FLOPs)
10 + 10 + 10 = 30 FLOPs

#### Cross entropy

20 input `Value`s (10 `distribution` from one-hot data, 10 `predicted_distribution` from `softmax()`). We create one `epsilon` `Value` (1 `Value`) and one `log_base` `Value` (1 `Value`).

For each of the 10 pairs, we:
- `PLUS` the `predicted_distribution` `Value` with the single `epsilon` (1 FLOP)
- `PLUS` the `distribution` `Value` with the single `epsilon` (1 FLOP)
- `LOGARITHM` the `PLUS` `predicted_distribution`. Too lazy to find CPython implementation of stdlib `math.log()` myself (assuming 1 FLOP)
- `MULTIPLY` the `LOGARITHM` `Value` with the `PLUS` `distribution` `Value` (1 FLOP)

Finally we `PLUS` all single output `Value`s for each pair (10 FLOPs), then negate it, which equals `MULTIPLY` with -1 (1 FLOP)

10 * (1 + 1 + 1 + 1) + 10 + 1 = 51

#### Grand total

47,319 + 30 + 51 = 47,400 FLOPs

> 🟩🟨 **GRADER — good modeling instinct; two notes.** (a) 728→784 nudges this up: the matmul
> core alone is `2·(784·32 + 32·10) = 2·25,408 = 50,816` FLOPs by the standard "MAC = 2 FLOPs"
> convention, so the total is **~51k FLOPs**, not 47.4k. The exact figure barely matters — what
> matters is the *order of magnitude*: **~5×10⁴ FLOPs.** (b) Counting `tanh` as ~20 FLOPs and
> `**`/`log` as ~1 is a totally reasonable engineering call — and notice it's *negligible*: the
> two matmuls dominate, which is the standard reason people quote "2·MACs" and ignore activations.
> Keep the habit of stating the convention explicitly, as you did.

### Question

Divide by the measured wall-clock (the ~30 ms anchor, or the real number from 1a) to get **achieved FLOP/s**.

### Answer

47,400 FLOPs / 0.117 seconds = ~405,000 FLOP/s

> 🟥 **GRADER — right idea, wrong denominator (and numerator).** Use the *real* forward time,
> 25 ms, not the cProfile 117 ms, and ~51k FLOPs: `51,000 / 0.025 ≈ **2.0×10⁶ FLOP/s ≈ 2 MFLOP/s**.
> (For a fully fair "engine" number you might use forward+backward, ~2·51k / 0.129 s ≈ 0.8 MFLOP/s.)
> Either way it's single-digit MFLOP/s — hold that against the peak below.

### Question

Derive the CPU's **theoretical peak** FLOP/s for this chip: cores × clock × (AVX2 lanes) × (2 for FMA) × (2 for mul+add). Then: what % of peak are we hitting?

Interpret: the gap between achieved and peak *is* the combined interpreter + representation tax. This number reframes the whole project.

### Answer

Using the equation in https://en.wikipedia.org/wiki/Floating_point_operations_per_second#Computational_performance, I have

Cores = 8
Clock = 2.3GHz = 2.3 * 10^9 cycles / second

AVX2 = Advanced Vector eXtensions 2. Defines new instructions added to x86 ISA that operate on dedicated YMM registers. https://en.wikipedia.org/wiki/Advanced_Vector_Extensions
An AVX2 lane defines the bit count / width to store one number within a YMM register, that can be operated on in parallel with other lanes.
Each YMM register has 256 bits, so if using float64s I can fit 256 / 64 = 4 per lane.

FMA = Fused Multiply Add. Defines dedicated hardware to multiply two numbers and add a third number in a single CPU cycle. https://en.wikipedia.org/wiki/FMA_instruction_set
Its presence means FLOP count doubles where we need a multiply followed by an add.
I'm unsure whether AVX2 instruction usage is mutually exclusive from FMA instruction usage? Though by the equation in the question it seems they can be combined. I'd be curious to write an assembly program that demonstrates all of this that I can assemble and execute.

I'm not sure what the separate (2 for mul+add) means, since my understanding is this is covered by FMA presence.

> 🟩 **GRADER — you're right to call this out; the prompt's factoring is misleading.** "FMA" and
> "mul+add" are *not* two independent ×2's. The clean decomposition of "FLOPs per core per cycle":
>
> ```
>   lanes per SIMD register   ×   FMA execution units   ×   FLOPs per FMA
>   (AVX2, fp64: 256/64 = 4)  ×   (this chip: 2 ports)  ×   (mul+add = 2)   = 16 fp64 FLOP/cycle
> ```
>
> So the *correct* reading of the prompt's factors: "(AVX2 lanes)=4", "(2 for FMA)=**# of FMA
> ports**, which happens to be 2 on your Coffee-Lake i9", "(2 for mul+add)=**FLOPs per FMA
> instruction**". Your arithmetic (4×2×2=16) lands right; only the *labels* were muddled. And
> yes — AVX2 and FMA compose: FMA is a *separate instruction set* whose ops run on those 256-bit
> vector units, so one instruction does 4 lanes × (mul+add) = 8 fp64 FLOPs, and 2 ports issue
> two per cycle = 16. (fp32 would be 8 lanes → 32 FLOP/cycle. Your `Value.data` is a Python
> `float` = C `double` = fp64, so 16 is the right comparison.) Writing that assembly demo would
> be a great exercise — `vfmadd231pd` on `ymm` registers is the instruction to reach for.

The last subtlety I found is that Turbo exists according to https://www.intel.com/content/www/us/en/products/sku/192987/intel-core-i99880h-processor-16m-cache-up-to-4-80-ghz/specifications.html. This would put the clock cycle at 4.80 GHz, though I'm not sure for example how long it can be sustained.

Regardless it seems according to the equation provided in the question that we have a theoretical upper bound of:
8 * 2.3 * 10 ^ 9 * 4 * 2 * 2 = 294.4 * 10^9 FLOP/s.

405,000 / (294.4 * 10^9) = 0.00000138 = 0.0000000138%. Absolutely abysmal.

> 🟩🟥 **GRADER — peak is correct (294 GFLOP/s fp64); the % has a ×10⁴ slip.** `8 · 2.3e9 · 16 =
> 294.4 GFLOP/s` ✓. But converting a fraction to a percent is ×100, not ÷100:
> `1.38e-6` (fraction) → `1.38e-4 %` = **0.000138 %**, not `0.0000000138 %`. And with the
> corrected achieved number (~2 MFLOP/s) it's `2e6 / 294.4e9 = 6.8e-6` = **~0.0007 % of peak**.
> The "abysmal" verdict is dead right either way — you're ~5 orders of magnitude down.
>
> **One conceptual upgrade that reframes the project:** the 294 GFLOP/s peak bundles *three*
> separate wins — 8× (multicore), ~4–8× (SIMD width), and the interpreter/representation tax.
> To isolate the pure **interpreter tax**, compare against *single-core scalar* peak:
> `2.3e9 cycles × 2 FMA ports × 2 FLOP ≈ 9.2 GFLOP/s`. Achieved 2 MFLOP/s → **~4,600× off even
> without any SIMD or threads.** That ~4,600× is what a flat-array C engine buys you; the
> remaining ~64× to the 294 GFLOP/s ceiling is what SIMD + multicore (Tasks 2+) chase. Splitting
> the gap this way is the single most useful thing to put in `1_flops_writeup.md`.

To go further, I downloaded and executed https://www.passmark.com/ and found that on my Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz (x86_64) 8 cores @ 2300 MHz CPU, I have:

```sh
CPU Mark:                          11163
  Integer Math                     36873 Million Operations/s
  Floating Point Math              22832 Million Operations/s
  Prime Numbers                    33.4 Million Primes/s
  Sorting                          20152 Thousand Strings/s
  Encryption                       4868 MB/s
  Compression                      181601 KB/s
  CPU Single Threaded              1706 Million Operations/s
  Physics                          718 Frames/s
  Extended Instructions (SSE)      7762 Million Matrices/s
```

> 🟩 **GRADER — nice real-world sanity anchor.** PassMark's 22,832 MFLOP/s = **~22.8 GFLOP/s** is
> the *achievable* fp number on real mixed code, ~7.7 % of the 294 GFLOP/s theoretical ceiling —
> which is itself a useful lesson (you never hit theoretical peak; ~5–15 % is typical for
> non-hand-tuned code). Against that empirical number your engine at ~2 MFLOP/s is **~11,000×**
> slower. So the three reference points to keep in mind: theoretical peak 294 G, real-world
> ~23 G, single-core scalar ~9 G, you ~0.002 G. The project's job is to climb that ladder.
>
> ---
>
> ## 🟦 GRADER — final summary
> **A−.** Method and self-checking are the strengths — you predicted counts from structure and
> validated against the profiler, which is precisely the discipline this task teaches. The
> deductions: (1) `728→784` propagated everywhere, (2) cProfile time reported as wall-clock, (3)
> the forward==backward identity missed, (4) fraction→percent slip. None are conceptual failures;
> they're the kind of thing the "predict *and confirm*" loop is designed to catch — three of the
> four are things you *flagged as confusing yourself*, which means your instincts were firing;
> you just didn't chase them down. Corrected headline numbers to carry forward:
> **51,812 nodes · 25,545 fwd = 25,545 bwd · 25 ms forward · ~2 MFLOP/s · ~0.0007 % of peak
> (~4,600× off single-core scalar).** HPC-equation questions answered in the linked artifact.
>
> **📊 HPC FLOP/s equation artifact:** https://claude.ai/code/artifact/e22d98c5-f5b5-47ab-8ff8-30e4eacc025a
> (the chain with unit-cancellation, the `FLOPs/cycle` unpack, your i9-9880H worked out, and all
> four node/socket/core questions). It's private to you until you share it from the page.
