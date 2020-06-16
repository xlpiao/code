/**
 * File              : cuda_conv.cu
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2020.06.16
 * Last Modified Date: 2020.06.16
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 * NOTE:             : cuda convolution (conv1d(), conv2d())
 */
#include <stdio.h>
#include <iostream>

#define INPUT_SIZE 8
#define KERNEL_SIZE 5
#define STRIDE 2
#define PADDING 2
#define OUTPUT_SIZE ((INPUT_SIZE + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1)

__constant__ float kernel[KERNEL_SIZE];
__constant__ unsigned int stride;
__constant__ unsigned int padding;

__global__ void conv1d(float* input, float* output) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int k = 0; k < KERNEL_SIZE; k++) {
    int col = gid * STRIDE - PADDING + k;
    if (col >= 0 && col < INPUT_SIZE) {
      output[gid] += input[col] * kernel[k];
    }
  }
}

void initData(float* data, const unsigned int size) {
  for (int i = 0; i < size; i++) {
    data[i] = 1;
  }
}

void print1d(float* data, const unsigned int size) {
  for (int i = 0; i < size; i++) {
    std::cout << data[i] << ", ";
  }
  std::cout << std::endl;
}

int main(void) {
  dim3 block_dim(0);
  dim3 grid_dim(0);

  //// 1D convolution
  std::cout << "\n--- 1D convolution ---\n" << std::endl;
  float* h_input = NULL;
  h_input = (float*)malloc(INPUT_SIZE * sizeof(float));
  initData(h_input, INPUT_SIZE);

  std::cout << "input: " << std::endl;
  print1d(h_input, INPUT_SIZE);
  std::cout << std::endl;

  float* d_input = NULL;
  cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(float));
  cudaMemcpy(d_input, h_input, INPUT_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);

  float h_kernel[KERNEL_SIZE] = {2, 2, 2, 2, 2};
  std::cout << "kernel: " << std::endl;
  print1d(h_kernel, KERNEL_SIZE);
  std::cout << std::endl;
  cudaMemcpyToSymbol(kernel, &h_kernel, sizeof(kernel));

  const unsigned int h_stride = STRIDE;
  cudaMemcpyToSymbol(stride, &h_stride, sizeof(stride));

  const unsigned int h_padding = PADDING;
  cudaMemcpyToSymbol(padding, &h_padding, sizeof(padding));

  block_dim.x = 4;
  grid_dim.x = OUTPUT_SIZE / block_dim.x;

  float* d_output = NULL;
  cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(float));
  cudaMemset(d_output, 0, OUTPUT_SIZE * sizeof(float));

  conv1d<<<grid_dim, block_dim>>>(d_input, d_output);
  cudaDeviceSynchronize();

  float* h_output = NULL;
  h_output = (float*)calloc(OUTPUT_SIZE, sizeof(float));
  cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout << "output: " << std::endl;
  print1d(h_output, OUTPUT_SIZE);

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
