#include <stdio.h>

// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>

__global__ void print(int* input) {
  // printf("gridDim.x : %d , gridDim.y : %d , gridDim.z : %d \n", gridDim.x,
  // gridDim.y, gridDim.z);
  // printf("blockDim.x : %d , blockDim.y : %d , blockDim.z : %d \n",
  // blockDim.x, blockDim.y, blockDim.z);
  printf("threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n", threadIdx.x,
         threadIdx.y, threadIdx.z);
  printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d \n", blockIdx.x,
         blockIdx.y, blockIdx.z);
}

__global__ void print_details(int* input) {
  /*int gid =
    (threadIdx.z * blockDim.x * blockDim.y) +
    (threadIdx.y * blockDim.x) +
    (threadIdx.x) +
    (blockDim.x * blockDim.y * blockDim.z * blockIdx.x) +
    (blockDim.x * blockDim.y * blockDim.z * gridDim.x * blockIdx.y) +
    (blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y *
    blockIdx.z);*/

  int tid = (threadIdx.z * blockDim.x * blockDim.y) +
            (threadIdx.y * blockDim.x) + threadIdx.x;
  int num_of_thread_in_a_block = blockDim.x * blockDim.y * blockDim.z;

  int block_offset = num_of_thread_in_a_block * blockIdx.x;

  int num_of_threads_in_a_row = num_of_thread_in_a_block * gridDim.x;
  int row_offset = num_of_threads_in_a_row * blockIdx.y;

  int num_of_thread_in_xy = num_of_thread_in_a_block * gridDim.x * gridDim.y;
  int z_offset = num_of_thread_in_xy * blockIdx.z;

  int gid = tid + block_offset + row_offset + z_offset;

  printf("tid : %d , gid : %d , value : %d \n", tid, gid, input[gid]);
}

int main() {
  int x = 4;
  int y = 16;
  int z = 32;
  int size = x * y * z;

  int byte_size = size * sizeof(int);

  int* h_input;
  h_input = (int*)malloc(byte_size);

  time_t t;
  srand((unsigned)time(&t));

  for (int i = 0; i < size; i++) {
    h_input[i] = (int)(rand() & 0xff);
  }

  int* d_input;
  cudaMalloc((void**)&d_input, byte_size);

  cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

  dim3 block_dim(2, 8, 16);
  dim3 grid_dim(x / block_dim.x, y / block_dim.y, z / block_dim.z);

  // print_details <<< grid_dim, block_dim >>> (d_input);
  print<<<grid_dim, block_dim>>>(d_input);

  cudaDeviceSynchronize();

  cudaFree(d_input);
  free(h_input);

  cudaDeviceReset();
}
