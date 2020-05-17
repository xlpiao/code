#include <stdio.h>

__global__ void print1D(int* input) {
  int x_axis = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("%d\n", x_axis);

  int gid = x_axis;

  input[gid] = gid;

  printf("gridDim(%d), blockDim(%d), blockIdx(%d), threadIdx(%d), gid: %d\n",
         gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, gid);
}

__global__ void print2D(int* input) {
  int x_axis = blockIdx.x * blockDim.x + threadIdx.x;  // col
  int y_axis = blockIdx.y * blockDim.y + threadIdx.y;  // row
  // printf("%d, %d\n", x_axis, y_axis);

  int gid = y_axis * gridDim.x * blockDim.x + x_axis;

  input[gid] = gid;

  printf(
      "gridDim(%d,%d), blockDim(%d,%d), blockIdx(%d,%d), threadIdx(%d,%d), "
      "gid: %d\n",
      gridDim.x, gridDim.y, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y,
      threadIdx.x, threadIdx.y, gid);
}

__global__ void print3D(int* input) {
  int x_axis = blockIdx.x * blockDim.x + threadIdx.x;
  int y_axis = blockIdx.y * blockDim.y + threadIdx.y;
  int z_axis = blockIdx.z * blockDim.z + threadIdx.z;
  // printf("%d, %d, %d\n", x_axis, y_axis, z_axis);

  int gid = z_axis * blockDim.x * gridDim.x * blockDim.y * gridDim.y +
            y_axis * blockDim.x * gridDim.x + x_axis;

  input[gid] = gid;

  printf(
      "gridDim(%d,%d,%d), blockDim(%d,%d,%d), blockIdx(%d,%d,%d), "
      "threadIdx(%d,%d,%d), gid: %d\n",
      gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
      blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,
      gid);
}

int main(void) {
  dim3 size(2, 4, 8);
  dim3 block_dim(0);
  dim3 grid_dim(0);

  int* h_input = NULL;
  int* d_input = NULL;

  //// 1D
  printf("\nprint 1D:\n");
  h_input = (int*)calloc(size.x, sizeof(int));
  cudaMalloc((void**)&d_input, size.x * sizeof(int));
  // cudaMemcpy(d_input, h_input, size.x * sizeof(int), cudaMemcpyHostToDevice);

  block_dim.x = 2;
  grid_dim.x = size.x / block_dim.x;

  print1D<<<grid_dim, block_dim>>>(d_input);
  cudaDeviceSynchronize();

  // cudaMemcpy(h_input, d_input, size.x * sizeof(int), cudaMemcpyDeviceToHost);
  free(h_input);
  cudaFree(d_input);

  //// 2D
  printf("\nprint 2D:\n");
  h_input = (int*)calloc(size.x * size.y, sizeof(int));
  cudaMalloc((void**)&d_input, size.x * size.y * sizeof(int));

  block_dim.y = 4;
  grid_dim.y = size.y / block_dim.y;

  print2D<<<grid_dim, block_dim>>>(d_input);
  cudaDeviceSynchronize();

  // cudaMemcpy(h_input, d_input, size.x * size.y * sizeof(int),
  // cudaMemcpyDeviceToHost);
  free(h_input);
  cudaFree(d_input);

  //// 3D
  printf("\nprint 3D:\n");
  h_input = (int*)calloc(size.x * size.y * size.z, sizeof(int));
  cudaMalloc((void**)&d_input, size.x * size.y * size.z * sizeof(int));

  block_dim.z = 8;
  grid_dim.z = size.z / block_dim.z;

  print3D<<<grid_dim, block_dim>>>(d_input);
  cudaDeviceSynchronize();

  // cudaMemcpy(h_input, d_input, size.x * size.y * size.z * sizeof(int),
  // cudaMemcpyDeviceToHost);
  free(h_input);
  cudaFree(d_input);

  //// reset
  cudaDeviceReset();
}
