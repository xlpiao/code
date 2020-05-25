#include <stdio.h>

__global__ void no_divergence(int* input, dim3 size) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size.x) {
    float a = 0.0;
    int warp_id = gid / 32;

    if (warp_id % 2 == 0) {
      a = 100.0;
      printf("warp(%d), a(%.0f)\n", warp_id, a);

    } else {
      a = 200.0;
      printf("warp(%d), a(%.0f)\n", warp_id, a);
    }
  }
}

__global__ void divergence(int* input, dim3 size) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size.x) {
    float a = 0.0;

    if (gid % 2 == 0) {
      a = 100.0;
      printf("warp(%d), a(%.0f)\n", gid, a);

    } else {
      a = 200.0;
      printf("warp(%d), a(%.0f)\n", gid, a);
    }
  }
}

void initInput(int* input, int size) {
  for (int index = 0; index < size; index++) {
    input[index] = index;
  }
}

int main(void) {
  dim3 size(32, 0, 0);
  dim3 block_dim(0);
  dim3 grid_dim(0);

  int* h_input = NULL;
  int* d_input = NULL;

  h_input = (int*)calloc(size.x, sizeof(int));
  initInput(h_input, size.x);
  cudaMalloc((void**)&d_input, size.x * sizeof(int));
  cudaMemcpy(d_input, h_input, size.x * sizeof(int), cudaMemcpyHostToDevice);

  block_dim.x = 32;
  grid_dim.x = size.x / block_dim.x + 1;

  printf("\nno warp divergence occurred:\n");
  no_divergence<<<grid_dim, block_dim>>>(d_input, size);
  cudaDeviceSynchronize();

  printf("\nwarp divergence occurred:\n");
  divergence<<<grid_dim, block_dim>>>(d_input, size);
  cudaDeviceSynchronize();

  cudaFree(d_input);
  free(h_input);

  //// reset
  cudaDeviceReset();
}
