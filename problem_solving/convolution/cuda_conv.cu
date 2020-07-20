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
__global__ void conv1d_naive(float* input, float* output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float temp = 0.0f;
  for (int k = 0; k < KERNEL_SIZE; k++) {
    int col_offset = col * STRIDE - PADDING + k;
    if (col_offset >= 0 && col_offset < INPUT_SIZE) {
      temp += input[col_offset] * kernel[k];
    }
  }

  output[col] = temp;
}

//// shared memory
#define BLOCK_SIZE 4
__global__ void conv1d_shared(float* input, float* output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float shared_block[BLOCK_SIZE];

  float temp = 0.0f;
  for (int k = 0; k < KERNEL_SIZE; k++) {
    int col_offset = col * STRIDE - PADDING + k;

    shared_block[threadIdx.x] = input[col_offset];
    __syncthreads();

    if (col_offset >= 0 && col_offset < INPUT_SIZE) {
      temp += shared_block[threadIdx.x] * kernel[k];
    }
  }

  output[col] = temp;
}

void initData(float* data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = i + 1;
  }
}

void print(float* data, dim3 dim) {
  for (int x = 0; x < dim.x; x++) {
    if (dim.y == 0) {
      std::cout << data[x] << ",  ";
    } else {
      for (int y = 0; y < dim.y; y++) {
        std::cout << data[x * dim.x + y] << ",  ";
      }
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
}

//// constant memory
__constant__ float kernel2d[KERNEL_SIZE * KERNEL_SIZE];

//// global memory
__global__ void conv2d_naive(float* input, float* output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float temp = 0.0f;
  for (int m = 0; m < KERNEL_SIZE; m++) {
    for (int n = 0; n < KERNEL_SIZE; n++) {
      int col_offset = col * STRIDE - PADDING + m;
      int row_offset = row * STRIDE - PADDING + n;
      if ((col_offset >= 0 && col_offset < INPUT_SIZE) &&
          (row_offset >= 0 && row_offset < INPUT_SIZE)) {
        temp += input[row_offset * INPUT_SIZE + col_offset] *
                kernel2d[m * KERNEL_SIZE + n];
      }
    }

    output[row * OUTPUT_SIZE + col] = temp;
  }
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
  initData(h_input, INPUT_SIZE);
  print(h_input, dim3(INPUT_SIZE, 0, 0));
  std::cout << std::endl;

  cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(float));
  cudaMemcpy(d_input, h_input, INPUT_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);

  float h_kernel[KERNEL_SIZE] = {1, 2, 4, 2, 1};
  std::cout << "kernel: " << std::endl;
  print(h_kernel, dim3(KERNEL_SIZE, 0, 0));
  std::cout << std::endl;
  cudaMemcpyToSymbol(kernel, &h_kernel, sizeof(kernel));

  block_dim.x = BLOCK_SIZE;
  grid_dim.x = OUTPUT_SIZE / block_dim.x;

  cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(float));
  h_output = (float*)calloc(OUTPUT_SIZE, sizeof(float));

  //// using global memory
  cudaMemset(d_output, 0, OUTPUT_SIZE * sizeof(float));
  conv1d_naive<<<grid_dim, block_dim>>>(d_input, d_output);
  cudaDeviceSynchronize();

  memset(h_output, 0, OUTPUT_SIZE);
  cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout << "output: " << std::endl;
  print(h_output, dim3(OUTPUT_SIZE, 0, 0));
  std::cout << std::endl;

  //// using shared memory
  cudaMemset(d_output, 0, OUTPUT_SIZE * sizeof(float));
  conv1d_shared<<<grid_dim, block_dim>>>(d_input, d_output);
  cudaDeviceSynchronize();

  memset(h_output, 0, OUTPUT_SIZE);
  cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout << "output: " << std::endl;
  print(h_output, dim3(OUTPUT_SIZE, 0, 0));
  std::cout << std::endl;

  //// free
  cudaFree(d_input);
  cudaFree(d_output);
  free(h_input);
  free(h_output);

  //// 2D convolution
  std::cout << "\n--- 2D convolution ---\n" << std::endl;
  h_input = (float*)calloc(INPUT_SIZE * INPUT_SIZE, sizeof(float));
  initData(h_input, INPUT_SIZE * INPUT_SIZE);
  std::cout << "input: " << std::endl;
  print(h_input, dim3(INPUT_SIZE, INPUT_SIZE, 0));
  std::cout << std::endl;

  cudaMalloc((void**)&d_input, INPUT_SIZE * INPUT_SIZE * sizeof(float));
  cudaMemcpy(d_input, h_input, INPUT_SIZE * INPUT_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);

  float h_kernel2d[KERNEL_SIZE * KERNEL_SIZE] = {1, 1, 1, 1, 1,   //
                                                 1, 2, 2, 2, 1,   //
                                                 1, 2, 4, 2, 1,   //
                                                 1, 2, 2, 2, 1,   //
                                                 1, 1, 1, 1, 1};  //
  std::cout << "kernel: " << std::endl;
  print(h_kernel2d, dim3(KERNEL_SIZE, KERNEL_SIZE, 0));
  std::cout << std::endl;
  cudaMemcpyToSymbol(kernel2d, &h_kernel2d, sizeof(kernel2d));

  block_dim.x = BLOCK_SIZE;
  block_dim.y = BLOCK_SIZE;
  grid_dim.x = OUTPUT_SIZE / block_dim.x;
  grid_dim.y = OUTPUT_SIZE / block_dim.y;

  cudaMalloc((void**)&d_output, OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float));
  h_output = (float*)calloc(OUTPUT_SIZE * OUTPUT_SIZE, sizeof(float));
  cudaMemset(d_output, 0, OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float));

  conv2d_naive<<<grid_dim, block_dim>>>(d_input, d_output);
  cudaDeviceSynchronize();

  memset(h_output, 0, OUTPUT_SIZE * OUTPUT_SIZE);
  cudaMemcpy(h_output, d_output, OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout << "output: " << std::endl;
  print(h_output, dim3(OUTPUT_SIZE, OUTPUT_SIZE, 0));
  std::cout << std::endl;

  //// free
  cudaFree(d_input);
  cudaFree(d_output);
  free(h_input);
  free(h_output);

  //// reset
  cudaDeviceReset();
}
