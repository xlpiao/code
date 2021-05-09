/**
 * File              : cpu_conv2d.cc
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2020.06.15
 * Last Modified Date: 2021.05.09
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */

/* NOTE: convolutional output size */
/* output_size = (input_size + 2 * padding - kernel_size) / stride + 1 */
/* input_index = (output_index * stride - padding) + kernel_index * dilation */

#include <torch/extension.h>

#include <iostream>
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
  // const auto ifm_size = ifm_batch * ifm_channel * ifm_height * ifm_width;

  float *wgt_p = (float *)wgt.data_ptr();
  auto wgt_a = wgt.accessor<float, 4>();
  const auto wgt_batch = wgt_a.size(0);
  const auto wgt_channel = wgt_a.size(1);
  const auto wgt_height = wgt_a.size(2);
  const auto wgt_width = wgt_a.size(3);
  // const auto wgt_size = wgt_batch * wgt_channel * wgt_height * wgt_width;
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
  // const auto ofm_size = ofm_batch * ofm_channel * ofm_height * ofm_width;

  for (int ofm_b = 0; ofm_b < ofm_batch; ofm_b++) {
    for (int ofm_c = 0; ofm_c < ofm_channel; ofm_c++) {
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
                  ofm_p[ofm_idx] += ifm_p[ifm_idx] * wgt_p[wgt_idx];
                }
              }
            }
          }
          ofm_p[ofm_idx] += bias_p[ofm_c];
        }
      }
    }
  }

  return ofm;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2d", &conv2d, "naive conv2d with one cpu core");
}
