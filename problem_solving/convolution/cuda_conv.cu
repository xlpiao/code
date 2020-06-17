/**
 * File              : cuda_conv.cu
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2020.06.16
 * Last Modified Date: 2020.06.16
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 * NOTE:             : cuda convolution (conv1d(), conv2d())
 */

#include <iostream>

#define INPUT_SIZE 8
#define KERNEL_SIZE 5
#define STRIDE 2
#define PADDING 2  // For same input/output size PADDING = KERNEL_SIZE / 2
#define OUTPUT_SIZE ((INPUT_SIZE + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1)

//// constant memory
__constant__ float kernel[KERNEL_SIZE];

//// global memory
__global__ void conv1d_global(float* input, float* output) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int k = 0; k < KERNEL_SIZE; k++) {
    int col = gid * STRIDE - PADDING + k;
    if (col >= 0 && col < INPUT_SIZE) {
      output[gid] += input[col] * kernel[k];
    }
  }
}

//// shared memory
#define BLOCK_SIZE 4
__global__ void conv1d_shared(float* input, float* output) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float shared_block[BLOCK_SIZE];

  for (int k = 0; k < KERNEL_SIZE; k++) {
    int col = gid * STRIDE - PADDING + k;

    shared_block[threadIdx.x] = input[col];
    __syncthreads();
    if (col >= 0 && col < INPUT_SIZE) {
      output[gid] += shared_block[threadIdx.x] * kernel[k];
    }
  }
}

void initData(float* data, int size, float value) {
  for (int i = 0; i < size; i++) {
    data[i] = i + 1;
  }
}

void print1d(float* data, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << data[i] << ", ";
  }
  std::cout << std::endl;
}

int main(void) {
  dim3 block_dim(0);
  dim3 grid_dim(0);

  float* h_input = NULL;
  float* d_input = NULL;
  float* d_output = NULL;
  float* h_output = NULL;

  //// 1D convolution
  std::cout << "\n--- 1D convolution ---\n" << std::endl;
  std::cout << "input: " << std::endl;
  h_input = (float*)malloc(INPUT_SIZE * sizeof(float));
  initData(h_input, INPUT_SIZE, 1.0);
  print1d(h_input, INPUT_SIZE);
  std::cout << std::endl;

  cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(float));
  cudaMemcpy(d_input, h_input, INPUT_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);

  float h_kernel[KERNEL_SIZE] = {1, 2, 4, 2, 1};
  std::cout << "kernel: " << std::endl;
  print1d(h_kernel, KERNEL_SIZE);
  std::cout << std::endl;
  cudaMemcpyToSymbol(kernel, &h_kernel, sizeof(kernel));

  block_dim.x = BLOCK_SIZE;
  grid_dim.x = OUTPUT_SIZE / block_dim.x;

  cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(float));
  h_output = (float*)calloc(OUTPUT_SIZE, sizeof(float));

  //// using global memory
  cudaMemset(d_output, 0, OUTPUT_SIZE * sizeof(float));
  conv1d_global<<<grid_dim, block_dim>>>(d_input, d_output);
  cudaDeviceSynchronize();

  memset(h_output, 0, OUTPUT_SIZE);
  cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout << "output: " << std::endl;
  print1d(h_output, OUTPUT_SIZE);
  std::cout << std::endl;

  //// using shared memory
  cudaMemset(d_output, 0, OUTPUT_SIZE * sizeof(float));
  conv1d_shared<<<grid_dim, block_dim>>>(d_input, d_output);
  cudaDeviceSynchronize();

  memset(h_output, 0, OUTPUT_SIZE);
  cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout << "output: " << std::endl;
  print1d(h_output, OUTPUT_SIZE);
  std::cout << std::endl;

  cudaFree(d_input);
  cudaFree(d_output);
  free(h_input);
  free(h_output);

  //// 2D convolution
  // printf("\nprint 2D:\n");
  // h_input = (float*)calloc(INPUT_SIZE * size.y, sizeof(float));
  // initInput(h_input, INPUT_SIZE * size.y);
  // cudaMalloc((void**)&input, INPUT_SIZE * size.y * sizeof(float));
  // cudaMemcpy(input, h_input, INPUT_SIZE * size.y * sizeof(float),
  // cudaMemcpyHostToDevice);

  // block_dim.y = 4;
  // grid_dim.y = size.y / block_dim.y;

  // print2D<<<grid_dim, block_dim>>>(d_input);
  // cudaDeviceSynchronize();

  // cudaMemcpy(h_input, d_input, INPUT_SIZE * size.y * sizeof(float),
  // cudaMemcpyDeviceToHost);
  // cudaFree(d_input);
  // free(h_input);

  //// reset
  cudaDeviceReset();
}
