#include <stdio.h>
#include <stdlib.h>

inline void check_cuda_errors(const char *filename, const int line_number) {
#ifdef DEBUG
  cudaThreadSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error at %s:%i: %s\n", filename, line_number,
           cudaGetErrorString(error));
    exit(-1);
  }
#endif
}

__global__ void foo(int *ptr) { *ptr = 7; }

int main(void) {
  foo<<<1, 1>>>(0);
  check_cuda_errors(__FILE__, __LINE__);

  return 0;
}
