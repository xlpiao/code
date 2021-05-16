/**
 * File              : gpu_conv2d.cu
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2020.06.16
 * Last Modified Date: 2020.07.31
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 * NOTE:             : cuda conv2d
 */

#include <torch/extension.h>

#include <iostream>

#define CONV2D conv2d_optimized
#define UNFOLD unfold_optimized

__global__ void cuda_conv2d_naive(float *ofm,
                                  float *ifm,
                                  float *wgt,
                                  float *bias,
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
                                  const unsigned int bias_size,
                                  const unsigned int stride,
                                  const unsigned int padding,
                                  const unsigned int dilation,
                                  const unsigned int groups) {
  int ofm_b = blockIdx.x * blockDim.x + threadIdx.x;
  int ofm_c = blockIdx.y * blockDim.y + threadIdx.y;

  // for (int ofm_b = 0; ofm_b < ofm_batch; ofm_b++) {
  // for (int ofm_c = 0; ofm_c < ofm_channel; ofm_c++) {
  if (ofm_b >= 0 && ofm_b < ofm_batch) {
    if (ofm_c >= 0 && ofm_c < ofm_channel) {
      for (int ofm_h = 0; ofm_h < ofm_height; ofm_h++) {
        for (int ofm_w = 0; ofm_w < ofm_width; ofm_w++) {
          float temp = 0.0f;
          int ofm_idx = ofm_b * ofm_channel * ofm_height * ofm_width +
                        ofm_c * ofm_height * ofm_width + ofm_h * ofm_width +
                        ofm_w;
          for (int wgt_b = ofm_c, wgt_c = 0; wgt_c < wgt_channel; wgt_c++) {
            for (int wgt_h = 0; wgt_h < wgt_height; wgt_h++) {
              for (int wgt_w = 0; wgt_w < wgt_width; wgt_w++) {
                int ifm_b = ofm_b;
                int ifm_c = wgt_c;
                int ifm_h = (ofm_h * stride - padding) + wgt_h * dilation;
                int ifm_w = (ofm_w * stride - padding) + wgt_w * dilation;
                if ((ifm_h >= 0 && ifm_h < ifm_height) &&
                    (ifm_w >= 0 && ifm_w < ifm_width)) {
                  int ifm_idx = ifm_b * ifm_channel * ifm_height * ifm_width +
                                ifm_c * ifm_height * ifm_width +
                                ifm_h * ifm_width + ifm_w;
                  int wgt_idx = wgt_b * wgt_channel * wgt_height * wgt_width +
                                wgt_c * wgt_height * wgt_width +
                                wgt_h * wgt_width + wgt_w;
                  temp += ifm[ifm_idx] * wgt[wgt_idx];
                }
              }
            }
          }
          temp += bias[ofm_c];
          ofm[ofm_idx] = temp;
        }
      }
    }
  }
}

void conv2d_naive(float *ifm_p,
                  const unsigned int ifm_batch,
                  const unsigned int ifm_channel,
                  const unsigned int ifm_height,
                  const unsigned int ifm_width,
                  const unsigned int ifm_size,
                  float *wgt_p,
                  const unsigned int wgt_batch,
                  const unsigned int wgt_channel,
                  const unsigned int wgt_height,
                  const unsigned int wgt_width,
                  const unsigned int wgt_size,
                  float *bias_p,
                  const unsigned int bias_size,
                  float *ofm_p,
                  const unsigned int ofm_batch,
                  const unsigned int ofm_channel,
                  const unsigned int ofm_height,
                  const unsigned int ofm_width,
                  const unsigned int ofm_size,
                  unsigned int stride,
                  unsigned int padding,
                  unsigned int dilation,
                  unsigned int groups) {
  float *ifm_d, *wgt_d, *bias_d, *ofm_d;
  cudaMalloc(&ifm_d, ifm_size * sizeof(float));
  cudaMalloc(&wgt_d, wgt_size * sizeof(float));
  cudaMalloc(&bias_d, bias_size * sizeof(float));
  cudaMalloc(&ofm_d, ofm_size * sizeof(float));

  cudaMemcpy(ifm_d, ifm_p, ifm_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(wgt_d, wgt_p, wgt_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(bias_d, bias_p, bias_size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block(0);  // blockDim: # of threads
  block.x = 1;
  block.y = 1;

  dim3 grid(0);  // gridDim: # of blocks
  grid.x = (ofm_batch + block.x - 1) / block.x;
  grid.y = (ofm_channel + block.y - 1) / block.y;

  std::cout << "block(x,y,z): "
            << "(" << block.x << "," << block.y << "," << block.z << ")"
            << std::endl;
  std::cout << "grid(x,y,z): "
            << "(" << grid.x << "," << grid.y << "," << grid.z << ")"
            << std::endl;

  cuda_conv2d_naive<<<grid, block>>>(ofm_d,
                                     ifm_d,
                                     wgt_d,
                                     bias_d,
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
                                     bias_size,
                                     stride,
                                     padding,
                                     dilation,
                                     groups);

  // sync and get output
  cudaDeviceSynchronize();
  cudaMemcpy(ofm_p, ofm_d, ofm_size * sizeof(float), cudaMemcpyDeviceToHost);

  // free
  cudaFree(ifm_d);
  cudaFree(wgt_d);
  cudaFree(bias_d);
  cudaFree(ofm_d);
}

__global__ void cuda_conv2d_stream(float *ofm,
                                   float *ifm,
                                   float *wgt,
                                   float *bias,
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
                                   const unsigned int bias_size,
                                   const unsigned int stride,
                                   const unsigned int padding,
                                   const unsigned int dilation,
                                   const unsigned int groups) {
  int ofm_w = blockIdx.x * blockDim.x + threadIdx.x;
  int ofm_h = blockIdx.y * blockDim.y + threadIdx.y;

  // __shared__ float input[ifm_channel][ifm_height][ifm_width];
  // __shared__ float kernel[wgt_channel][wgt_height][wgt_width];

  __shared__ float input[3][32][32];
  __shared__ float kernel[3][3][3];

  float temp = 0.0f;
  for (int wgt_c = 0; wgt_c < wgt_channel; wgt_c++) {
    for (int wgt_h = 0; wgt_h < wgt_height; wgt_h++) {
      for (int wgt_w = 0; wgt_w < wgt_width; wgt_w++) {
        int ifm_c = wgt_c;
        int ifm_h = (ofm_h * stride - padding) + wgt_h * dilation;
        int ifm_w = (ofm_w * stride - padding) + wgt_w * dilation;
        if ((ifm_c >= 0 && ifm_c < ifm_channel) &&
            (ifm_h >= 0 && ifm_h < ifm_height) &&
            (ifm_w >= 0 && ifm_w < ifm_width)) {
          int ifm_idx =
              ifm_c * ifm_height * ifm_width + ifm_h * ifm_width + ifm_w;
          int wgt_idx =
              wgt_c * wgt_height * wgt_width + wgt_h * wgt_width + wgt_w;
          input[ifm_c][ifm_h][ifm_w] = ifm[ifm_idx];
          kernel[wgt_c][wgt_h][wgt_w] = wgt[wgt_idx];
          __syncthreads();
        }
        if ((ifm_c >= 0 && ifm_c < ifm_channel) &&
            (ifm_h >= 0 && ifm_h < ifm_height) &&
            (ifm_w >= 0 && ifm_w < ifm_width)) {
          temp += input[ifm_c][ifm_h][ifm_w] * kernel[wgt_c][wgt_h][wgt_w];
        }
      }
    }
  }
  ofm[ofm_h * ofm_width + ofm_w] = temp;
}

void conv2d_stream(float *ifm_p,
                   const unsigned int ifm_batch,
                   const unsigned int ifm_channel,
                   const unsigned int ifm_height,
                   const unsigned int ifm_width,
                   const unsigned int ifm_size,
                   float *wgt_p,
                   const unsigned int wgt_batch,
                   const unsigned int wgt_channel,
                   const unsigned int wgt_height,
                   const unsigned int wgt_width,
                   const unsigned int wgt_size,
                   float *bias_p,
                   const unsigned int bias_size,
                   float *ofm_p,
                   const unsigned int ofm_batch,
                   const unsigned int ofm_channel,
                   const unsigned int ofm_height,
                   const unsigned int ofm_width,
                   const unsigned int ofm_size,
                   unsigned int stride,
                   unsigned int padding,
                   unsigned int dilation,
                   unsigned int groups) {
  /* create cuda stream */
  cudaStream_t stream[ifm_batch];
  for (int i = 0; i < ifm_batch; i++) {
    cudaStreamCreate(&stream[i]);
  }

  dim3 block(0);  // blockDim: # of threads
  block.x = ofm_width;
  block.y = ofm_height;
  // block.z = ifm_channel;

  dim3 grid(0);  // gridDim: # of blocks
  grid.x = (ofm_width + block.x - 1) / block.x;
  grid.y = (ofm_height + block.y - 1) / block.y;
  // grid.y = (ifm_channel + block.z - 1) / block.z;

  std::cout << "block(x,y,z): "
            << "(" << block.x << "," << block.y << "," << block.z << ")"
            << std::endl;
  std::cout << "grid(x,y,z): "
            << "(" << grid.x << "," << grid.y << "," << grid.z << ")"
            << std::endl;

  int ifm_chunkSize = ifm_size / ifm_batch;
  int wgt_chunkSize = wgt_size / wgt_batch;
  int bias_chunkSize = bias_size / ofm_batch;
  int ofm_chunkSize = ofm_size / ifm_batch / wgt_batch;

  float *ifm_d[ifm_batch], *wgt_d[wgt_batch], *bias_d[ofm_batch],
      *ofm_d[ofm_batch * ofm_channel];
  for (int i = 0; i < ifm_batch; i++) {
    cudaMallocHost(&ifm_d[i], ifm_chunkSize * sizeof(float));
    cudaMallocHost(&bias_d[i], bias_chunkSize * sizeof(float));
  }
  for (int j = 0; j < wgt_batch; j++) {
    cudaMallocHost(&wgt_d[j], wgt_chunkSize * sizeof(float));
  }
  for (int k = 0; k < ofm_batch * ofm_channel; k++) {
    cudaMallocHost(&ofm_d[k], ofm_chunkSize * sizeof(float));
  }

  for (int i = 0; i < ifm_batch; i++) {
    int ifm_offset = ifm_chunkSize * i;

    cudaMemcpyAsync(ifm_d[i],
                    ifm_p + ifm_offset,
                    sizeof(float) * ifm_chunkSize,
                    cudaMemcpyHostToDevice,
                    stream[i]);
    for (int j = 0; j < wgt_batch; j++) {
      int wgt_offset = wgt_chunkSize * j;
      cudaMemcpyAsync(wgt_d[j],
                      wgt_p + wgt_offset,
                      sizeof(float) * wgt_chunkSize,
                      cudaMemcpyHostToDevice,
                      stream[i]);
      int ofm_offset = ofm_chunkSize * (i * wgt_batch + j);
      cuda_conv2d_stream<<<grid, block>>>(ofm_d[i * wgt_batch + j],
                                          ifm_d[i],
                                          wgt_d[j],
                                          bias_d[i],
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
                                          bias_size,
                                          stride,
                                          padding,
                                          dilation,
                                          groups);
      cudaMemcpyAsync(ofm_p + ofm_offset,
                      ofm_d[i * wgt_batch + j],
                      ofm_chunkSize * sizeof(float),
                      cudaMemcpyDeviceToHost,
                      stream[i]);
    }
  }
  cudaDeviceSynchronize();

  /* destroy cuda stream */
  for (int i = 0; i < ifm_batch; i++) {
    cudaStreamDestroy(stream[i]);
  }
}

__global__ void cuda_im2col(float *ifm_p,
                            const unsigned int ifm_batch,
                            const unsigned int ifm_channel,
                            const unsigned int ifm_height,
                            const unsigned int ifm_width,
                            const unsigned int ifm_size,
                            float *wgt_p,
                            const unsigned int wgt_batch,
                            const unsigned int wgt_channel,
                            const unsigned int wgt_height,
                            const unsigned int wgt_width,
                            const unsigned int wgt_size,
                            float *bias_p,
                            const unsigned int bias_size,
                            float *ofm_p,
                            const unsigned int ofm_batch,
                            const unsigned int ofm_channel,
                            const unsigned int ofm_height,
                            const unsigned int ofm_width,
                            const unsigned int ofm_size,
                            unsigned int stride,
                            unsigned int padding,
                            unsigned int dilation,
                            unsigned int groups,
                            float *ifm_im2col,
                            unsigned int ifm_im2col_size) {
  __shared__ float ifmShared[1024];
  int ofm_b = blockIdx.z * blockDim.z + threadIdx.z;
  int wgt_s = blockIdx.y * blockDim.y + threadIdx.y;
  int ofm_s = blockIdx.x * blockDim.x + threadIdx.x;
  const int wgt_im2col_size = wgt_channel * wgt_height * wgt_width;
  const int channel_im2col_size = ofm_height * ofm_width;

  if (ofm_b >= 0 && ofm_b < ofm_batch && wgt_s >= 0 &&
      wgt_s < wgt_im2col_size && ofm_s >= 0 && ofm_s < channel_im2col_size) {
    const int wgt_c = wgt_s / wgt_height / wgt_width;
    const int wgt_h = (wgt_s / wgt_width) % wgt_height;
    const int wgt_w = wgt_s % wgt_width;
    const int ofm_h = (ofm_s / ofm_width) % ofm_height;
    const int ofm_w = ofm_s % ofm_width;
    int ifm_im2col_idx = ofm_b * wgt_im2col_size * channel_im2col_size +
                         wgt_s * channel_im2col_size + ofm_s;
    int ifm_b = ofm_b;
    int ifm_c = wgt_c;
    int ifm_h = (ofm_h * stride - padding) + wgt_h * dilation;
    int ifm_w = (ofm_w * stride - padding) + wgt_w * dilation;
    if ((ifm_h >= 0 && ifm_h < ifm_height) &&
        (ifm_w >= 0 && ifm_w < ifm_width)) {
      int ifm_idx = ifm_b * ifm_channel * ifm_height * ifm_width +
                    ifm_c * ifm_height * ifm_width + ifm_h * ifm_width + ifm_w;
      // printf("im2col(%d, %d)\n", ifm_idx, threadIdx.x);
      ifmShared[threadIdx.x] = ifm_p[ifm_idx];
      __syncthreads();
      ifm_im2col[ifm_im2col_idx] = ifmShared[threadIdx.x];
    }
  }
}

__global__ void cuda_gemm(float *ifm_p,
                          const unsigned int ifm_batch,
                          const unsigned int ifm_channel,
                          const unsigned int ifm_height,
                          const unsigned int ifm_width,
                          const unsigned int ifm_size,
                          float *wgt_p,
                          const unsigned int wgt_batch,
                          const unsigned int wgt_channel,
                          const unsigned int wgt_height,
                          const unsigned int wgt_width,
                          const unsigned int wgt_size,
                          float *bias_p,
                          const unsigned int bias_size,
                          float *ofm_p,
                          const unsigned int ofm_batch,
                          const unsigned int ofm_channel,
                          const unsigned int ofm_height,
                          const unsigned int ofm_width,
                          const unsigned int ofm_size,
                          unsigned int stride,
                          unsigned int padding,
                          unsigned int dilation,
                          unsigned int groups,
                          float *ifm_im2col,
                          unsigned int ifm_im2col_size) {
  __shared__ float ifmShared[1024];
  int ofm_b = blockIdx.z * blockDim.z + threadIdx.z;
  int ofm_c = blockIdx.y * blockDim.y + threadIdx.y;
  int ofm_s = blockIdx.x * blockDim.x + threadIdx.x;
  const int wgt_im2col_size = wgt_channel * wgt_height * wgt_width;
  const int channel_im2col_size = ofm_height * ofm_width;
  if (ofm_b >= 0 && ofm_b < ofm_batch && ofm_c >= 0 && ofm_c < ofm_channel &&
      ofm_s >= 0 && ofm_s < channel_im2col_size) {
    float temp = 0.0f;
    const int ofm_h = (ofm_s / ofm_width) % ofm_height;
    const int ofm_w = ofm_s % ofm_width;
    for (int wgt_s = 0; wgt_s < wgt_im2col_size; wgt_s++) {
      const int wgt_c = wgt_s / wgt_height / wgt_width;
      const int wgt_h = (wgt_s / wgt_width) % wgt_height;
      const int wgt_w = wgt_s % wgt_width;
      int ifm_im2col_idx = ofm_b * wgt_im2col_size * channel_im2col_size +
                           wgt_s * channel_im2col_size + ofm_s;
      ifmShared[threadIdx.x] = ifm_im2col[ifm_im2col_idx];
      __syncthreads();
      // printf("gemm(%d, %d)\n", ifm_im2col_idx, threadIdx.x);
      temp += ifmShared[threadIdx.x] * wgt_p[ofm_c * wgt_im2col_size + wgt_s];
    }
    int ofm_idx = ofm_b * ofm_channel * channel_im2col_size +
                  ofm_c * channel_im2col_size + ofm_s;
    ofm_p[ofm_idx] = temp;
  }
}

void conv2d_optimized(float *ifm_p,
                      const unsigned int ifm_batch,
                      const unsigned int ifm_channel,
                      const unsigned int ifm_height,
                      const unsigned int ifm_width,
                      const unsigned int ifm_size,
                      float *wgt_p,
                      const unsigned int wgt_batch,
                      const unsigned int wgt_channel,
                      const unsigned int wgt_height,
                      const unsigned int wgt_width,
                      const unsigned int wgt_size,
                      float *bias_p,
                      const unsigned int bias_size,
                      float *ofm_p,
                      const unsigned int ofm_batch,
                      const unsigned int ofm_channel,
                      const unsigned int ofm_height,
                      const unsigned int ofm_width,
                      const unsigned int ofm_size,
                      unsigned int stride,
                      unsigned int padding,
                      unsigned int dilation,
                      unsigned int groups) {
  float *ifm_d, *wgt_d, *bias_d, *ofm_d;
  cudaMalloc(&ifm_d, ifm_size * sizeof(float));
  cudaMalloc(&wgt_d, wgt_size * sizeof(float));
  cudaMalloc(&bias_d, bias_size * sizeof(float));
  cudaMalloc(&ofm_d, ofm_size * sizeof(float));

  cudaMemcpy(ifm_d, ifm_p, ifm_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(wgt_d, wgt_p, wgt_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(bias_d, bias_p, bias_size * sizeof(float), cudaMemcpyHostToDevice);

  int ifm_im2col_size =
      ofm_batch * wgt_channel * wgt_height * wgt_width * ofm_height * ofm_width;
  float *ifm_im2col;
  cudaMalloc(&ifm_im2col, ifm_im2col_size * sizeof(float));
  cudaMemset(&ifm_im2col, 0, ifm_im2col_size * sizeof(float));

  dim3 block(0);  // blockDim: # of threads
  dim3 grid(0);   // gridDim: # of blocks

  const int wgt_im2col_size = wgt_channel * wgt_height * wgt_width;
  const int channel_im2col_size = ofm_height * ofm_width;
  block.x = channel_im2col_size;
  block.y = 1;
  block.z = 1;
  grid.x = (channel_im2col_size + block.x - 1) / block.x;
  grid.y = (wgt_im2col_size + block.y - 1) / block.y;
  grid.z = (ofm_batch + block.z - 1) / block.z;
  std::cout << "block(x,y,z): "
            << "(" << block.x << "," << block.y << "," << block.z << ")"
            << std::endl;
  std::cout << "grid(x,y,z): "
            << "(" << grid.x << "," << grid.y << "," << grid.z << ")"
            << std::endl;

  cuda_im2col<<<grid, block>>>(ifm_d,
                               ifm_batch,
                               ifm_channel,
                               ifm_height,
                               ifm_width,
                               ifm_size,
                               wgt_d,
                               wgt_batch,
                               wgt_channel,
                               wgt_height,
                               wgt_width,
                               wgt_size,
                               bias_d,
                               bias_size,
                               ofm_d,
                               ofm_batch,
                               ofm_channel,
                               ofm_height,
                               ofm_width,
                               ofm_size,
                               stride,
                               padding,
                               dilation,
                               groups,
                               ifm_im2col,
                               ifm_im2col_size);

  block.x = channel_im2col_size;
  block.y = 1;
  block.z = 1;
  grid.x = (channel_im2col_size + block.x - 1) / block.x;
  grid.y = (ofm_channel + block.y - 1) / block.y;
  grid.z = (ofm_batch + block.z - 1) / block.z;
  std::cout << "block(x,y,z): "
            << "(" << block.x << "," << block.y << "," << block.z << ")"
            << std::endl;
  std::cout << "grid(x,y,z): "
            << "(" << grid.x << "," << grid.y << "," << grid.z << ")"
            << std::endl;

  cuda_gemm<<<grid, block>>>(ifm_d,
                             ifm_batch,
                             ifm_channel,
                             ifm_height,
                             ifm_width,
                             ifm_size,
                             wgt_d,
                             wgt_batch,
                             wgt_channel,
                             wgt_height,
                             wgt_width,
                             wgt_size,
                             bias_d,
                             bias_size,
                             ofm_d,
                             ofm_batch,
                             ofm_channel,
                             ofm_height,
                             ofm_width,
                             ofm_size,
                             stride,
                             padding,
                             dilation,
                             groups,
                             ifm_im2col,
                             ifm_im2col_size);

  // sync and get output
  cudaDeviceSynchronize();
  cudaMemcpy(ofm_p, ofm_d, ofm_size * sizeof(float), cudaMemcpyDeviceToHost);

  // free
  cudaFree(ifm_d);
  cudaFree(wgt_d);
  cudaFree(bias_d);
  cudaFree(ofm_d);
  cudaFree(ifm_im2col);
}

torch::Tensor conv2d(torch::Tensor &ifm,
                     torch::Tensor &wgt,
                     torch::Tensor &bias,
                     unsigned int stride,
                     unsigned int padding,
                     unsigned int dilation,
                     unsigned int groups) {
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
  assert(wgt_channel == ifm_channel);

  float *bias_p = (float *)bias.data_ptr();
  auto bias_a = bias.accessor<float, 1>();
  const auto bias_size = bias_a.size(0);
  assert(bias_size == wgt_batch);

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

  CONV2D(ifm_p,
         ifm_batch,
         ifm_channel,
         ifm_height,
         ifm_width,
         ifm_size,
         wgt_p,
         wgt_batch,
         wgt_channel,
         wgt_height,
         wgt_width,
         wgt_size,
         bias_p,
         bias_size,
         ofm_p,
         ofm_batch,
         ofm_channel,
         ofm_height,
         ofm_width,
         ofm_size,
         stride,
         padding,
         dilation,
         groups);

  return ofm;
}

__global__ void cuda_unfold_optimized(float *ofm,
                                      float *ifm,
                                      float *wgt,
                                      float *bias,
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
                                      const unsigned int bias_size,
                                      const unsigned int stride,
                                      const unsigned int padding,
                                      const unsigned int dilation,
                                      const unsigned int groups) {
  int ofm_b = blockIdx.z * blockDim.z + threadIdx.z;
  int wgt_s = blockIdx.y * blockDim.y + threadIdx.y;
  int ofm_s = blockIdx.x * blockDim.x + threadIdx.x;

  int ofm_size = ofm_height * ofm_width;
  int wgt_size = wgt_channel * wgt_height * wgt_width;

  if (ofm_b >= 0 && ofm_b < ofm_batch && wgt_s >= 0 && wgt_s < wgt_size &&
      ofm_s >= 0 && ofm_s < ofm_size) {
    int wgt_c = wgt_s / wgt_height / wgt_width;
    int wgt_h = (wgt_s / wgt_width) % wgt_height;
    int wgt_w = wgt_s % wgt_width;
    int ofm_h = (ofm_s / ofm_width) % ofm_height;
    int ofm_w = ofm_s % ofm_width;
    int ofm_idx = ofm_b * wgt_size * ofm_size + wgt_s * ofm_size + ofm_s;
    int ifm_b = ofm_b;
    int ifm_c = wgt_c;
    int ifm_h = (ofm_h * stride - padding) + wgt_h * dilation;
    int ifm_w = (ofm_w * stride - padding) + wgt_w * dilation;
    if ((ifm_h >= 0 && ifm_h < ifm_height) &&
        (ifm_w >= 0 && ifm_w < ifm_width)) {
      int ifm_idx = ifm_b * ifm_channel * ifm_height * ifm_width +
                    ifm_c * ifm_height * ifm_width + ifm_h * ifm_width + ifm_w;
      ofm[ofm_idx] = ifm[ifm_idx];
    }
  }
}

void unfold_optimized(float *ifm_p,
                      unsigned int ifm_batch,
                      unsigned int ifm_channel,
                      unsigned int ifm_height,
                      unsigned int ifm_width,
                      unsigned int ifm_size,
                      float *wgt_p,
                      unsigned int wgt_batch,
                      unsigned int wgt_channel,
                      unsigned int wgt_height,
                      unsigned int wgt_width,
                      unsigned int wgt_size,
                      float *bias_p,
                      unsigned int bias_size,
                      float *ofm_p,
                      unsigned int ofm_batch,
                      unsigned int ofm_channel,
                      unsigned int ofm_height,
                      unsigned int ofm_width,
                      unsigned int ofm_size,
                      unsigned int stride,
                      unsigned int padding,
                      unsigned int dilation,
                      unsigned int groups) {
  float *ifm_d, *wgt_d, *bias_d, *ofm_d;
  cudaMalloc(&ifm_d, ifm_size * sizeof(float));
  cudaMalloc(&wgt_d, wgt_size * sizeof(float));
  cudaMalloc(&bias_d, bias_size * sizeof(float));
  cudaMalloc(&ofm_d, ofm_size * sizeof(float));

  cudaMemcpy(ifm_d, ifm_p, ifm_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(wgt_d, wgt_p, wgt_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(bias_d, bias_p, bias_size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block(0);  // blockDim: # of threads
  block.x = 1;
  block.y = 1;
  block.z = 1;

  dim3 grid(0);  // gridDim: # of blocks
  grid.x = (ofm_width * ofm_height + block.x - 1) / block.x;
  grid.y = ((wgt_channel * wgt_height * wgt_width) + block.y - 1) / block.y;
  grid.z = (ofm_batch + block.z - 1) / block.z;

  std::cout << "block(x,y,z): "
            << "(" << block.x << "," << block.y << "," << block.z << ")"
            << std::endl;
  std::cout << "grid(x,y,z): "
            << "(" << grid.x << "," << grid.y << "," << grid.z << ")"
            << std::endl;

  ofm_size = ofm_batch * wgt_channel * wgt_height * wgt_width * ofm_height *
             ofm_height;
  cuda_unfold_optimized<<<grid, block>>>(ofm_d,
                                         ifm_d,
                                         wgt_d,
                                         bias_d,
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
                                         bias_size,
                                         stride,
                                         padding,
                                         dilation,
                                         groups);

  // sync and get output
  cudaDeviceSynchronize();
  cudaMemcpy(ofm_p, ofm_d, ofm_size * sizeof(float), cudaMemcpyDeviceToHost);

  // free
  cudaFree(ifm_d);
  cudaFree(wgt_d);
  cudaFree(bias_d);
  cudaFree(ofm_d);
}

torch::Tensor unfold(torch::Tensor &ifm,
                     torch::Tensor &wgt,
                     torch::Tensor &bias,
                     int stride,
                     int padding,
                     int dilation,
                     int groups) {
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
  assert(wgt_channel == ifm_channel);

  float *bias_p = (float *)bias.data_ptr();
  auto bias_a = bias.accessor<float, 1>();
  const auto bias_size = bias_a.size(0);
  assert(bias_size == wgt_batch);

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
  torch::Tensor ofm = torch::zeros({ofm_batch,
                                    wgt_channel * wgt_height * wgt_width,
                                    ofm_height * ofm_width});
  float *ofm_p = (float *)ofm.data_ptr();
  const auto ofm_size =
      ofm_batch * ofm_channel * ofm_height * ofm_width * wgt_width * wgt_height;

  UNFOLD(ifm_p,
         ifm_batch,
         ifm_channel,
         ifm_height,
         ifm_width,
         ifm_size,
         wgt_p,
         wgt_batch,
         wgt_channel,
         wgt_height,
         wgt_width,
         wgt_size,
         bias_p,
         bias_size,
         ofm_p,
         ofm_batch,
         ofm_channel,
         ofm_height,
         ofm_width,
         ofm_size,
         stride,
         padding,
         dilation,
         groups);

  return ofm;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2d", &conv2d, "naive conv2d with gpu cuda");
  m.def("unfold", &unfold, "naive unfold with gpu cuda");
}
