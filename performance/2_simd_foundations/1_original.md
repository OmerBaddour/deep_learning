# Task 2 — SIMD foundations on this hardware

**Prereq:** `general.md`. **Independent** of tasks 1/3/4 — can be done any time.
**Goal:** understand AVX2+FMA concretely enough to read and later write a kernel.
Reading + one tiny C toy; no changes to the library.

Concepts to nail down:
- **SIMD model:** one instruction, N lanes. Register widths: SSE=128b, AVX/AVX2=256b.
  256b / 32b = **8 float32 lanes**; 256b / 64b = **4 float64 lanes**.
- **This CPU's ISA extensions** (from `sysctl -n machdep.cpu.features
  machdep.cpu.leaf7_features`): SSE4.2, AVX, **AVX2**, **FMA**, F16C. Know what each
  adds. FMA = `a*b+c` in one instruction → *the* instruction for dot products/matmul.
- **Why FMA matters here:** a matmul inner loop is a sum of products; FMA does
  multiply + accumulate in one op with a single rounding → ~2× throughput + better
  numerics.
- **float32 vs float64:** 8 lanes vs 4. NNs use float32 — decide the fast path is
  float32 and understand the accuracy tradeoff.
- **Memory layout:** alignment, contiguous/row-major, AoS→SoA — why loads are cheap
  only when the data is laid out for them.
- **Roofline intuition:** is matmul at MNIST sizes compute-bound or memory/overhead-
  bound? Predict before task 6 measures.
- **How SIMD gets emitted:** (a) compiler auto-vectorization `-O3 -mavx2 -mfma`,
  (b) intrinsics (`<immintrin.h>`, `_mm256_fmadd_ps`), (c) hand asm (skip). Tasks 6
  uses (a) then (b).

**The toy (the "aha"):** a ~10-line C program summing a float array. Compile with and
without `-mavx2`, `objdump -d` both, and watch `vaddps` / `ymm` registers appear.

**Deliverable:** `2_simd_foundations_writeup.md` — lanes, FMA, float32 vs float64,
memory layout, roofline, and the objdump before/after.

## Session kickoff prompt
> "Read `data/omer_questions/simd_speedup_plan/general.md` and `2_simd_foundations.md`.
> Teach me AVX2+FMA on my i9-9880H. Walk me through a minimal C example where I can
> *see* SIMD instructions appear in the disassembly with `-mavx2`, then help me write
> `2_simd_foundations_writeup.md` covering lanes, FMA, float32 vs float64, memory
> layout, and the roofline idea."
