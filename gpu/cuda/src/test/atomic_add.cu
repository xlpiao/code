#include <stdio.h>
#include <stdlib.h>

__global__ void colonel(int *d_a) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  printf("Before %d, %d, %d, index = %d, *d_a = %d\n", blockIdx.x, blockDim.x,
         threadIdx.x, index, *d_a);
  atomicAdd(d_a, index);
  printf("After %d, %d, %d, index = %d, *d_a = %d\n", blockIdx.x, blockDim.x,
         threadIdx.x, index, *d_a);
}

int main() {
  int h_a = 0, *d_a;

  cudaMalloc((void **)&d_a, sizeof(int));
  cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);

  float elapsedTime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //// 1D
  colonel<<<4, 4>>>(d_a);  // global id: 0 ~ 15, atomicAdd = sum(0+1+2+3+...+15)

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("GPU Time elapsed: %f seconds\n", elapsedTime / 1000.0);

  cudaMemcpy(&h_a, d_a, sizeof(int), cudaMemcpyDeviceToHost);

  printf("h_a = %d\n", h_a);
  cudaFree(d_a);
}
