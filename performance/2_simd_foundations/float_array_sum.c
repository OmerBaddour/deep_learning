#include "stdio.h"
#include "stdlib.h"

#define ARRAY_LENGTH 256
#define AVX2_BYTES_PER_VECTOR 32

int main() {
  float array_1[ARRAY_LENGTH] = {1};
  float array_2[ARRAY_LENGTH] = {2};
  float array_result[ARRAY_LENGTH] = {0};
  
  int size_array_element = sizeof(array_1[0]);
  int num_lanes = AVX2_BYTES_PER_VECTOR / size_array_element;
  int num_vmovups_operations = ARRAY_LENGTH / num_lanes;
  printf("my system has AVX2 (%u bytes per vector)\n", AVX2_BYTES_PER_VECTOR);
  printf("size of each array element: %u\n", size_array_element);
  printf("we expect %u lanes, and %u vmovups operations\n\n", num_lanes, num_vmovups_operations);

  for (size_t i = 0; i < ARRAY_LENGTH; i++) {
    array_result[i] = array_1[i] * array_2[i];
  }
  printf("array result values: ");
  for (size_t i = 0; i < ARRAY_LENGTH; i++) {
    printf("%f, ", array_result[i]);
  }
  printf("\n");
  return 0;
}