#include <stdio.h>
#include <stdlib.h>

#define ARRAY_LENGTH 256

// -> vaddps on ymm registers
void add_arrays(
    const float *restrict a,
    const float *restrict b,
    float *restrict out
) {
  for (int i = 0; i < ARRAY_LENGTH; i++) {
    out[i] = a[i] + b[i];
  }
}

// -> vfmadd213ps: multiply and accumulate in one instruction
void fma_arrays(
    const float *restrict a,
    const float *restrict b,
    const float *restrict c,
    float *restrict out
) {
  for (int i = 0; i < ARRAY_LENGTH; i++) {
    out[i] = a[i] * b[i] + c[i];
  }
}

int main(void) {
  float a[ARRAY_LENGTH], b[ARRAY_LENGTH], c[ARRAY_LENGTH], out[ARRAY_LENGTH];
  for (int i = 0; i < ARRAY_LENGTH; i++) {
    a[i] = (float)rand() / RAND_MAX; // [0, 1]
    b[i] = (float)rand() / RAND_MAX;
    c[i] = (float)rand() / RAND_MAX;
  }

  add_arrays(a, b, out);
  printf("add: %f\n", out[0]);

  fma_arrays(a, b, c, out);
  printf("fma: %f\n", out[0]);
  return 0;
}
