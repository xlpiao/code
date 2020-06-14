#include <stdio.h>
#include <stdlib.h>

void initialize(int* input, int size) {
  for (int index = 0; index < size; index++) {
    input[index] = index;
  }
}

int cpu_reduction(int* input, int size) {
  int result = 0;
  for (int index = 0; index < size; index++) {
    result += index;
  }
  return result;
}

void showResult(int cpu_result, int gpu_result) {
  printf("cpu_result: %d, gpu_result: %d\n", cpu_result, gpu_result);
}

//// 1. reduction neighbored pairs kernel
__global__ void redunction_v1(int* input, int* output, int size) {
  int tid = threadIdx.x;
  int gid = blockDim.x * blockIdx.x + threadIdx.x;

  if (gid >= size) return;

  for (int offset = 1; offset <= blockDim.x / 2; offset *= 2) {
    if (tid % (2 * offset) == 0) {
      input[gid] += input[gid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = input[gid];
  }
}

//// 2. warp_divergence_improved of #1 reduction_v1
__global__ void reduction_v1_improved(int* input, int* output, int size) {
  int tid = threadIdx.x;
  int gid = blockDim.x * blockIdx.x + threadIdx.x;

  // local data block pointer
  int* i_data = input + blockDim.x * blockIdx.x;

  if (gid >= size) return;

  for (int offset = 1; offset <= blockDim.x / 2; offset *= 2) {
    int index = 2 * offset * tid;

    if (index < blockDim.x) {
      i_data[index] += i_data[index + offset];
    }

    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = input[gid];
  }
}

int main(int argc, char** argv) {
  // int size = 1 << 27;  // 128 Mb of data
  int size = 512;
  dim3 block(128);
  dim3 grid(size / block.x);

  int* h_ref = (int*)malloc(grid.x * sizeof(int));
  int gpu_result;

  //// input
  int* h_input;
  h_input = (int*)malloc(size * sizeof(int));
  initialize(h_input, size);

  //// cpu redunction
  int cpu_result = cpu_reduction(h_input, size);

  //// gpu redunction
  int *d_input, *d_output;
  cudaMalloc((void**)&d_input, size * sizeof(int));
  cudaMalloc((void**)&d_output, grid.x * sizeof(int));

  //// #1
  cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0, grid.x * sizeof(int));
  redunction_v1<<<grid, block>>>(d_input, d_output, size);
  cudaDeviceSynchronize();

  memset(h_ref, 0, grid.x * sizeof(int));
  cudaMemcpy(h_ref, d_output, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_result = 0;
  for (int i = 0; i < grid.x; i++) {
    gpu_result += h_ref[i];
  }
  showResult(cpu_result, gpu_result);

  //// #2
  cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0, grid.x * sizeof(int));
  reduction_v1_improved<<<grid, block>>>(d_input, d_output, size);
  cudaDeviceSynchronize();

  memset(h_ref, 0, grid.x * sizeof(int));
  cudaMemcpy(h_ref, d_output, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_result = 0;
  for (int i = 0; i < grid.x; i++) {
    gpu_result += h_ref[i];
  }
  showResult(cpu_result, gpu_result);

  cudaFree(d_output);
  cudaFree(d_input);
  free(h_ref);
  free(h_input);
  cudaDeviceReset();

  return 0;
}
