// Same kernels as float_array_sum.c, but hand-written with AVX2/FMA intrinsics
// instead of left to the auto-vectorizer. Requires -mavx2 -mfma.
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_LENGTH 256
#define LANES 8 // 32-byte ymm register / 4-byte float

void add_arrays(
    const float *restrict a,
    const float *restrict b,
    float *restrict out
) {
  for (int i = 0; i < ARRAY_LENGTH; i += LANES) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
  }
}

void fma_arrays(
    const float *restrict a,
    const float *restrict b,
    const float *restrict c,
    float *restrict out)
{
  for (int i = 0; i < ARRAY_LENGTH; i += LANES) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vc = _mm256_loadu_ps(c + i);
    // one instruction, one rounding: va * vb + vc
    _mm256_storeu_ps(out + i, _mm256_fmadd_ps(va, vb, vc));
  }
}

int main(void) {
  float a[ARRAY_LENGTH], b[ARRAY_LENGTH], c[ARRAY_LENGTH], out[ARRAY_LENGTH];
  for (int i = 0; i < ARRAY_LENGTH; i++) {
    a[i] = (float)rand() / RAND_MAX;
    b[i] = (float)rand() / RAND_MAX;
    c[i] = (float)rand() / RAND_MAX;
  }

  add_arrays(a, b, out);
  printf("add: %f\n", out[0]);

  fma_arrays(a, b, c, out);
  printf("fma: %f\n", out[0]);
  return 0;
}
