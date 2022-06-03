/**
 * File              : gpu_conv.cu
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2020.06.16
 * Last Modified Date: 2020.07.31
 * Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>
 * NOTE:             : cuda conv2d
 */

#include <iostream>

#define ifm_size 8
#define wgt_size 5
#define stride 2
#define padding 2  // For same ifm/ofm size padding = wgt_size / 2
#define ofm_size ((ifm_size + 2 * padding - wgt_size) / stride + 1)

//// constant memory
__constant__ float wgt[wgt_size];

//// global memory
__global__ void cuda_conv1d_naive(float* ifm, float* ofm) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float temp = 0.0f;
  for (int k = 0; k < wgt_size; k++) {
    int col_offset = col * stride - padding + k;
    if (col_offset >= 0 && col_offset < ifm_size) {
      temp += ifm[col_offset] * wgt[k];
    }
  }

  ofm[col] = temp;
}

//// shared memory
#define BLOCK_size 4
__global__ void cuda_conv1d_shared(float* ifm, float* ofm) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float shared_block[BLOCK_size];

  float temp = 0.0f;
  for (int k = 0; k < wgt_size; k++) {
    int col_offset = col * stride - padding + k;

    shared_block[threadIdx.x] = ifm[col_offset];
    __syncthreads();

    if (col_offset >= 0 && col_offset < ifm_size) {
      temp += shared_block[threadIdx.x] * wgt[k];
    }
  }

  ofm[col] = temp;
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
__constant__ float wgt2d[wgt_size * wgt_size];

//// global memory
__global__ void cuda_conv2d_naive(float* ifm, float* ofm) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float temp = 0.0f;
  for (int m = 0; m < wgt_size; m++) {
    for (int n = 0; n < wgt_size; n++) {
      int col_offset = col * stride - padding + m;
      int row_offset = row * stride - padding + n;
      if ((col_offset >= 0 && col_offset < ifm_size) &&
          (row_offset >= 0 && row_offset < ifm_size)) {
        temp +=
            ifm[row_offset * ifm_size + col_offset] * wgt2d[m * wgt_size + n];
      }
    }

    ofm[row * ofm_size + col] = temp;
  }
}

void test() {
  dim3 block(0);
  dim3 grid(0);

  float* h_ifm = NULL;
  float* d_ifm = NULL;
  float* d_ofm = NULL;
  float* h_ofm = NULL;

  //// 1D convolution
  std::cout << "\n--- 1D convolution ---\n" << std::endl;
  std::cout << "ifm: " << std::endl;
  h_ifm = (float*)malloc(ifm_size * sizeof(float));
  initData(h_ifm, ifm_size);
  print(h_ifm, dim3(ifm_size, 0, 0));
  std::cout << std::endl;

  cudaMalloc((void**)&d_ifm, ifm_size * sizeof(float));
  cudaMemcpy(d_ifm, h_ifm, ifm_size * sizeof(float), cudaMemcpyHostToDevice);

  float h_wgt[wgt_size] = {1, 2, 4, 2, 1};
  std::cout << "wgt: " << std::endl;
  print(h_wgt, dim3(wgt_size, 0, 0));
  std::cout << std::endl;
  cudaMemcpyToSymbol(wgt, &h_wgt, sizeof(wgt));

  block.x = BLOCK_size;
  grid.x = ofm_size / block.x;

  cudaMalloc((void**)&d_ofm, ofm_size * sizeof(float));
  h_ofm = (float*)calloc(ofm_size, sizeof(float));

  //// using global memory
  cudaMemset(d_ofm, 0, ofm_size * sizeof(float));
  cuda_conv1d_naive<<<grid, block>>>(d_ifm, d_ofm);
  cudaDeviceSynchronize();

  memset(h_ofm, 0, ofm_size);
  cudaMemcpy(h_ofm, d_ofm, ofm_size * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "ofm: " << std::endl;
  print(h_ofm, dim3(ofm_size, 0, 0));
  std::cout << std::endl;

  //// using shared memory
  cudaMemset(d_ofm, 0, ofm_size * sizeof(float));
  cuda_conv1d_shared<<<grid, block>>>(d_ifm, d_ofm);
  cudaDeviceSynchronize();

  memset(h_ofm, 0, ofm_size);
  cudaMemcpy(h_ofm, d_ofm, ofm_size * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "ofm: " << std::endl;
  print(h_ofm, dim3(ofm_size, 0, 0));
  std::cout << std::endl;

  //// free
  cudaFree(d_ifm);
  cudaFree(d_ofm);
  free(h_ifm);
  free(h_ofm);

  //// 2D convolution
  std::cout << "\n--- 2D convolution ---\n" << std::endl;
  h_ifm = (float*)calloc(ifm_size * ifm_size, sizeof(float));
  initData(h_ifm, ifm_size * ifm_size);
  std::cout << "ifm: " << std::endl;
  print(h_ifm, dim3(ifm_size, ifm_size, 0));
  std::cout << std::endl;

  cudaMalloc((void**)&d_ifm, ifm_size * ifm_size * sizeof(float));
  cudaMemcpy(d_ifm,
             h_ifm,
             ifm_size * ifm_size * sizeof(float),
             cudaMemcpyHostToDevice);

  float h_wgt2d[wgt_size * wgt_size] = {1, 1, 1, 1, 1,   //
                                        1, 2, 2, 2, 1,   //
                                        1, 2, 4, 2, 1,   //
                                        1, 2, 2, 2, 1,   //
                                        1, 1, 1, 1, 1};  //
  std::cout << "wgt: " << std::endl;
  print(h_wgt2d, dim3(wgt_size, wgt_size, 0));
  std::cout << std::endl;
  cudaMemcpyToSymbol(wgt2d, &h_wgt2d, sizeof(wgt2d));

  block.x = BLOCK_size;
  block.y = BLOCK_size;
  grid.x = ofm_size / block.x;
  grid.y = ofm_size / block.y;

  cudaMalloc((void**)&d_ofm, ofm_size * ofm_size * sizeof(float));
  h_ofm = (float*)calloc(ofm_size * ofm_size, sizeof(float));
  cudaMemset(d_ofm, 0, ofm_size * ofm_size * sizeof(float));

  cuda_conv2d_naive<<<grid, block>>>(d_ifm, d_ofm);
  cudaDeviceSynchronize();

  memset(h_ofm, 0, ofm_size * ofm_size);
  cudaMemcpy(h_ofm,
             d_ofm,
             ofm_size * ofm_size * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout << "ofm: " << std::endl;
  print(h_ofm, dim3(ofm_size, ofm_size, 0));
  std::cout << std::endl;

  //// free
  cudaFree(d_ifm);
  cudaFree(d_ofm);
  free(h_ifm);
  free(h_ofm);

  //// reset
  cudaDeviceReset();
}

int main(void) {
  test();
  return 0;
}
