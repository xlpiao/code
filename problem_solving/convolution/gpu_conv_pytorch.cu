/**
 * File              : gpu_conv.cu
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2020.06.16
 * Last Modified Date: 2020.07.31
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 * NOTE:             : cuda conv2d
 */

#include <iostream>

#if 0
#define ifm_size 8
#define wgt_size 5
#define stride 2
#define padding 2  // For same ifm/ofm size padding = wgt_size / 2
#define ofm_size ((ifm_size + 2 * padding - wgt_size) / stride + 1)

//// constant memory
__constant__ float wgt[wgt_size];

//// global memory
__global__ void conv1d_naive(float* ifm, float* ofm) {
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
__global__ void conv1d_shared(float* ifm, float* ofm) {
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
__global__ void conv2d_naive(float* ifm, float* ofm) {
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
  dim3 block_dim(0);
  dim3 grid_dim(0);

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

  block_dim.x = BLOCK_size;
  grid_dim.x = ofm_size / block_dim.x;

  cudaMalloc((void**)&d_ofm, ofm_size * sizeof(float));
  h_ofm = (float*)calloc(ofm_size, sizeof(float));

  //// using global memory
  cudaMemset(d_ofm, 0, ofm_size * sizeof(float));
  conv1d_naive<<<grid_dim, block_dim>>>(d_ifm, d_ofm);
  cudaDeviceSynchronize();

  memset(h_ofm, 0, ofm_size);
  cudaMemcpy(h_ofm, d_ofm, ofm_size * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "ofm: " << std::endl;
  print(h_ofm, dim3(ofm_size, 0, 0));
  std::cout << std::endl;

  //// using shared memory
  cudaMemset(d_ofm, 0, ofm_size * sizeof(float));
  conv1d_shared<<<grid_dim, block_dim>>>(d_ifm, d_ofm);
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

  block_dim.x = BLOCK_size;
  block_dim.y = BLOCK_size;
  grid_dim.x = ofm_size / block_dim.x;
  grid_dim.y = ofm_size / block_dim.y;

  cudaMalloc((void**)&d_ofm, ofm_size * ofm_size * sizeof(float));
  h_ofm = (float*)calloc(ofm_size * ofm_size, sizeof(float));
  cudaMemset(d_ofm, 0, ofm_size * ofm_size * sizeof(float));

  conv2d_naive<<<grid_dim, block_dim>>>(d_ifm, d_ofm);
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
#else
#include <torch/extension.h>
__global__ void cuda_conv2d_naive(float *ofm,
                                  float *ifm,
                                  float *wgt,
                                  const unsigned int ofm_batch,
                                  const unsigned int ofm_channel,
                                  const unsigned int ofm_height,
                                  const unsigned int ofm_width,
                                  const unsigned int ifm_batch,
                                  const unsigned int ifm_channel,
                                  const unsigned int ifm_height,
                                  const unsigned int ifm_width,
                                  const unsigned int wgt_batch,
                                  const unsigned int wgt_channel,
                                  const unsigned int wgt_height,
                                  const unsigned int wgt_width,
                                  const unsigned int stride,
                                  const unsigned int padding,
                                  const unsigned int dilation) {
#if ONEDIM
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int ofm_b = (gid / ofm_width / ofm_height / ofm_channel);
  int ofm_c = (gid / ofm_width / ofm_height) % ofm_channel;
  int ofm_h = (gid / ofm_width) % ofm_height;
  int ofm_w = (gid) % ofm_width;
#else
  int ofm_bc = blockIdx.x * blockDim.x + threadIdx.x;
  int ofm_b = ofm_bc / ofm_channel;
  int ofm_c = ofm_bc % ofm_channel;
  int ofm_h = blockIdx.z * blockDim.z + threadIdx.z;
  int ofm_w = blockIdx.y * blockDim.y + threadIdx.y;
#endif

  // printf("%d, %d, %d, %d\n", ofm_b, ofm_c, ofm_h, ofm_w);
  if (ofm_b >= 0 && ofm_b < ofm_batch) {
    if (ofm_c >= 0 && ofm_c < ofm_channel) {
      if (ofm_h >= 0 && ofm_h < ofm_height) {
        if (ofm_w >= 0 && ofm_w < ofm_width) {
  // for (int ofm_b = 0; ofm_b < ofm_batch; ofm_b++) {
    // for (int ofm_c = 0; ofm_c < ofm_channel; ofm_c++) {
      // for (int ofm_h = 0; ofm_h < ofm_height; ofm_h++) {
        // for (int ofm_w = 0; ofm_w < ofm_width; ofm_w++) {
          float sum = 0.0f;
          for (int wgt_b = ofm_c, wgt_c = 0; wgt_c < wgt_channel; wgt_c += 8) {
            for (int wgt_h = 0; wgt_h < wgt_height; wgt_h++) {
              for (int wgt_w = 0; wgt_w < wgt_width; wgt_w++) {
                int ifm_b = ofm_b;
                int ifm_c = wgt_c;
                int ifm_h = (ofm_h * stride - padding) + wgt_h * dilation;
                int ifm_w = (ofm_w * stride - padding) + wgt_w * dilation;
                if ((ifm_h >= 0 && ifm_h < ifm_height) &&
                    (ifm_w >= 0 && ifm_w < ifm_width)) {
                  for (int offset = 0; offset < 8; offset++) {
                  int ifm_idx = ifm_b * ifm_channel * ifm_height * ifm_width +
                                (ifm_c + offset) * ifm_height * ifm_width +
                                ifm_h * ifm_width + ifm_w;
                  int wgt_idx = wgt_b * wgt_channel * wgt_height * wgt_width +
                                (wgt_c + offset) * wgt_height * wgt_width +
                                wgt_h * wgt_width + wgt_w;
                  sum += ifm[ifm_idx] * wgt[wgt_idx];
                  }
                }
              }
            }
          }
          int ofm_idx = ofm_b * ofm_channel * ofm_height * ofm_width +
                        ofm_c * ofm_height * ofm_width +
                        ofm_h * ofm_width + ofm_w;
          ofm[ofm_idx] = sum;
        }
      }
    }
  }
}

torch::Tensor conv2d(torch::Tensor &ifm,
                     torch::Tensor &wgt,
                     unsigned int stride,
                     unsigned int padding,
                     unsigned int dilation) {
  float *ifm_p = (float *)ifm.data_ptr();
  auto ifm_a = ifm.accessor<float, 4>();
  const auto ifm_batch = ifm_a.size(0);
  const auto ifm_channel = ifm_a.size(1);
  const auto ifm_height = ifm_a.size(2);
  const auto ifm_width = ifm_a.size(3);
  const auto ifm_size = ifm_batch * ifm_channel * ifm_height * ifm_width;

  float *wgt_p = (float *)wgt.data_ptr();
  auto wgt_a = wgt.accessor<float, 4>();
  const auto wgt_batch = wgt_a.size(0);
  const auto wgt_channel = wgt_a.size(1);
  const auto wgt_height = wgt_a.size(2);
  const auto wgt_width = wgt_a.size(3);
  const auto wgt_size = wgt_batch * wgt_channel * wgt_height * wgt_width;
  assert(ifm_channel == wgt_channel);

  const auto ofm_batch = ifm_batch;
  const auto ofm_channel = wgt_batch;
  const auto ofm_height = ((ifm_height + 2 * padding - wgt_height) -
                           (wgt_height - 1) * (dilation - 1)) /
                              stride +
                          1;
  const auto ofm_width = ((ifm_width + 2 * padding - wgt_width) -
                          (wgt_width - 1) * (dilation - 1)) /
                             stride +
                         1;
  torch::Tensor ofm =
      torch::zeros({ofm_batch, ofm_channel, ofm_height, ofm_width});
  float *ofm_p = (float *)ofm.data_ptr();
  const auto ofm_size = ofm_batch * ofm_channel * ofm_height * ofm_width;

  float *ifm_d, *wgt_d, *ofm_d;
  cudaMalloc(&ifm_d, ifm_size * sizeof(float));
  cudaMalloc(&wgt_d, wgt_size * sizeof(float));
  cudaMalloc(&ofm_d, ofm_size * sizeof(float));

  cudaMemcpy(
      ifm_d, ifm_p, ifm_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(
      wgt_d, wgt_p, wgt_size * sizeof(float), cudaMemcpyHostToDevice);

#if ONEDIM
  dim3 block_dim(0);
  block_dim.x = 8;

  dim3 grid_dim(0);
  grid_dim.x = (ofm_size) / block_dim.x;
#else
  dim3 block_dim(0);
  block_dim.x = 1;
  block_dim.y = 1;
  block_dim.z = 1;

  dim3 grid_dim(0);
  grid_dim.x = (ofm_batch * ofm_channel) / block_dim.x;
  grid_dim.y = ofm_height / block_dim.y;
  grid_dim.z = ofm_width / block_dim.z;
#endif

  cuda_conv2d_naive<<<grid_dim, block_dim>>>(ofm_d,
                                             ifm_d,
                                             wgt_d,
                                             ofm_batch,
                                             ofm_channel,
                                             ofm_height,
                                             ofm_width,
                                             ifm_batch,
                                             ifm_channel,
                                             ifm_height,
                                             ifm_width,
                                             wgt_batch,
                                             wgt_channel,
                                             wgt_height,
                                             wgt_width,
                                             stride,
                                             padding,
                                             dilation);
  cudaDeviceSynchronize();

  cudaMemcpy(
      ofm_p, ofm_d, ofm_size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(ifm_d);
  cudaFree(wgt_d);
  cudaFree(ofm_d);

  return ofm;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2d", &conv2d, "naive conv2d with gpu cuda");
}
#endif
