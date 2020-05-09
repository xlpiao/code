#include <stdio.h>
#include <stdlib.h>

int main(void) {
  int *ptr = 0;

  // gimme!
  cudaError_t error = cudaMalloc((void **)&ptr, UINT_MAX);
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  return 0;
}
