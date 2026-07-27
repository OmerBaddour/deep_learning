# Task 1 — Instrument the current engine

**Prereq:** `general.md` (esp. "Why the current engine is slow").
**Goal:** replace hand-waving with real numbers about *where* time goes today, so the
later speedups can be attributed. Two separate exercises; each is its own session.

## 1a. Empirical profiling (a measurement session)

Using the 784→32→10 net and the training/eval loop from `notebooks/mnist.ipynb`:

### Question

Count objects: how many `Value` nodes are in the graph? How many `op.forward` /
`op.backward` calls happen per forward pass and per backward pass? (Instrument, or
reason from the graph structure and confirm.)

### Answer

#### Input layer

One time creation of 728 `Value` instances (1 per pixel)

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

---

### Question

Profile one forward pass and one forward+backward with `cProfile` (and/or a manual
timer). Where does wall-clock actually go — the graph traversal in `Value.forward`,
the `op.forward` dispatch, the `list`/`zip` comprehensions, the SGD update loop?

### Answer

Most of the `Value.forward()` time is spent in `_traverse()` and doing set operations with `_traverse()`'s `visited`.

```sh
✗ uv run python performance/1_instrument_current_engine/answer_benchmark_forward_pass.py profile
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

I'm also seeing that a non-trivial proportion of the time is spent doing `len()`, which is purely for correctness assertions. Perhaps there is a clever way to support a separate `test` mode that is separate to the `run_fast` mode or something. Though I am surprised `len()` takes more time than list `append()`, since under the hood I would think `len()` is a heap variable read and `append()` can provoke something time consuming like `realloc()` in C...

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
✗ uv run python performance/1_instrument_current_engine/answer_benchmark_forward_and_backward_pass.py profile 
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

## 1b. Theoretical FLOP/s from first principles (a separate session)

This one assumes **no prior comfort** with FLOP counting or the roofline model — the
point is to build the reasoning from scratch. Work through, in order:

### Question

What counts as a FLOP? Why is a multiply-accumulate usually counted as 2.

### Answer

After reading some of https://en.wikipedia.org/wiki/Floating_point_operations_per_second, a FLOP is a floating point operation. Multiply-accumulate is usually counted as 2 as with commonly available CPUs, this entails two floating point operations. It is however possible to create specific hardware that makes this a single operation.

Further unrelated questions based on my reading:
- can you explain the general FLOPS equation for HPC systems? more specifically:
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

### Question

Divide by the measured wall-clock (the ~30 ms anchor, or the real number from 1a) to get **achieved FLOP/s**.

### Answer

47,400 FLOPs / 0.117 seconds = ~405,000 FLOP/s

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

The last subtlety I found is that Turbo exists according to https://www.intel.com/content/www/us/en/products/sku/192987/intel-core-i99880h-processor-16m-cache-up-to-4-80-ghz/specifications.html. This would put the clock cycle at 4.80 GHz, though I'm not sure for example how long it can be sustained.

Regardless it seems according to the equation provided in the question that we have a theoretical upper bound of:
8 * 2.3 * 10 ^ 9 * 4 * 2 * 2 = 294.4 * 10^9 FLOP/s.

405,000 / (294.4 * 10^9) = 0.00000138 = 0.0000000138%. Absolutely abysmal.

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
