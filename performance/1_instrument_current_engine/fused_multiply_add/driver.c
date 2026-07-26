// driver.c — calls the asm kernel, measures wall-clock, reports GFLOP/s.
// clock_gettime docs: https://man7.org/linux/man-pages/man2/clock_gettime.2.html
#include <stdio.h>
#include <stdint.h>
#include <time.h>

// Implemented in fma_kernel.s. Returns a scalar so the loop isn't optimized away.
extern double fma_kernel(uint64_t iters);

int main(void) {
  // ~2 billion loop iterations keeps the kernel running ~1s — long enough
  // to swamp timer noise, short enough to stay in single-core turbo.
  uint64_t iters = 2000000000ULL;

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  double sink = fma_kernel(iters);       // the timed region
  clock_gettime(CLOCK_MONOTONIC, &t1);

  double secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
  // 10 accumulators * 4 fp64 lanes * 2 FLOPs(mul+add) = 80 FLOPs per iteration
  double flops = (double)iters * 80.0;

  printf("ignore: %g\n", sink);   // forces 'sink' to be used
  printf("time:   %.4f s\n", secs);
  printf("GFLOP/s (1 core): %.1f\n", flops / secs / 1e9);
  return 0;
}
