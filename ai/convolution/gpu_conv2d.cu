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
                  ofm[ofm_idx] += ifm[ifm_idx] * wgt[wgt_idx];
                }
              }
            }
          }
          ofm[ofm_idx] += bias[ofm_c];
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
          temp += ifm[ifm_idx] * wgt[wgt_idx];
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
  block.x = 1;
  block.y = 1;
  // block.z = ifm_channel;

  dim3 grid(0);  // gridDim: # of blocks
  grid.x = (ifm_width + block.x - 1) / block.x;
  grid.y = (ifm_height + block.y - 1) / block.y;
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

  // for (int i = 0; i < 32 * 32; i++) {
  // std::cout << ofm_p[i] << std::endl;
  // }
  /* destroy cuda stream */
  for (int i = 0; i < ifm_batch; i++) {
    cudaStreamDestroy(stream[i]);
  }
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

  conv2d_stream(ifm_p,
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

#if 0
  conv2d_naive(ifm_p,
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
#endif
  return ofm;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2d", &conv2d, "naive conv2d with gpu cuda");
}
