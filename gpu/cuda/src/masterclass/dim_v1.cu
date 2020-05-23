#include <stdio.h>

__global__ void print1D(int* input, dim3 size) {
  int axis_x = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("%d, %d\n", axis_x, size.x);

  int gid = axis_x;

  // input[gid] = gid;

  if (axis_x < size.x) {
    printf(
        "size(%d), blockDim(%d), blockIdx(%d), threadIdx(%d), input(%d), "
        "gid(%d)\n",
        size.x, blockDim.x, blockIdx.x, threadIdx.x, input[gid], gid);
  }
}

__global__ void print2D(int* input, dim3 size) {
  int axis_x = blockIdx.x * blockDim.x + threadIdx.x;  // col
  int axis_y = blockIdx.y * blockDim.y + threadIdx.y;  // row
  // printf("%d, %d, %d, %d\n", axis_x, axis_y, size.x, size.y);

  int gid = axis_y * size.x + axis_x;

  // input[gid] = gid;

  if (axis_x < size.x && axis_y < size.y) {
    printf(
        "size(%d,%d), blockDim(%d,%d), blockIdx(%d,%d), threadIdx(%d,%d), "
        "input(%d), gid(%d)\n",
        size.x, size.y, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y,
        threadIdx.x, threadIdx.y, input[gid], gid);
  }
}

__global__ void print3D(int* input, dim3 size) {
  int axis_x = blockIdx.x * blockDim.x + threadIdx.x;
  int axis_y = blockIdx.y * blockDim.y + threadIdx.y;
  int axis_z = blockIdx.z * blockDim.z + threadIdx.z;
  // printf("%d, %d, %d, %d, %d, %d\n", axis_x, axis_y, axis_z, size.x, size.y,
  // size.z);

  int gid = axis_z * size.x * size.y + axis_y * size.x + axis_x;

  // input[gid] = gid;

  if (axis_x < size.x && axis_y < size.y && axis_z < size.z) {
    printf(
        "size(%d,%d,%d), blockDim(%d,%d,%d), blockIdx(%d,%d,%d), "
        "threadIdx(%d,%d,%d), input(%d), gid(%d)\n",
        size.x, size.y, size.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x,
        blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,
        input[gid], gid);
  }
}

void initInput(int* input, int size) {
  for (int index = 0; index < size; index++) {
    input[index] = index;
  }
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
  initInput(h_input, size.x);
  cudaMalloc((void**)&d_input, size.x * sizeof(int));
  cudaMemcpy(d_input, h_input, size.x * sizeof(int), cudaMemcpyHostToDevice);

  block_dim.x = 2;
  grid_dim.x = size.x / block_dim.x + 1;

  print1D<<<grid_dim, block_dim>>>(d_input, size);
  cudaDeviceSynchronize();

  cudaMemcpy(h_input, d_input, size.x * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  free(h_input);

  //// 2D
  printf("\nprint 2D:\n");
  h_input = (int*)calloc(size.x * size.y, sizeof(int));
  initInput(h_input, size.x * size.y);
  cudaMalloc((void**)&d_input, size.x * size.y * sizeof(int));
  cudaMemcpy(d_input, h_input, size.x * size.y * sizeof(int),
             cudaMemcpyHostToDevice);

  block_dim.y = 4;
  grid_dim.y = size.y / block_dim.y + 1;

  print2D<<<grid_dim, block_dim>>>(d_input, size);
  cudaDeviceSynchronize();

  cudaMemcpy(h_input, d_input, size.x * size.y * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  free(h_input);

  //// 3D
  printf("\nprint 3D:\n");
  h_input = (int*)calloc(size.x * size.y * size.z, sizeof(int));
  initInput(h_input, size.x * size.y * size.z);
  cudaMalloc((void**)&d_input, size.x * size.y * size.z * sizeof(int));
  cudaMemcpy(d_input, h_input, size.x * size.y * size.z * sizeof(int),
             cudaMemcpyHostToDevice);

  block_dim.z = 8;
  grid_dim.z = size.z / block_dim.z + 1;

  print3D<<<grid_dim, block_dim>>>(d_input, size);
  cudaDeviceSynchronize();

  cudaMemcpy(h_input, d_input, size.x * size.y * size.z * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  free(h_input);

  //// reset
  cudaDeviceReset();
}
