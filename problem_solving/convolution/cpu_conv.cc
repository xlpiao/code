/**
 * File              : cpu_conv.cc
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2020.06.15
 * Last Modified Date: 2020.07.29
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */

/* NOTE: convolutional output size */
/* output_size = (input_size + 2 * padding - kernel_size) / stride + 1 */
/* input_index = (output_index * stride - padding) + kernel_index * dilation */

#include <iostream>

#if 0
#include <vector>

using Array1D = std::vector<float>;
using Array2D = std::vector<Array1D>;
using Array3D = std::vector<Array2D>;

class Functional {
public:
  Array1D conv1d(Array1D& ifm,
                 Array1D& wgt,
                 unsigned int stride = 1,
                 unsigned int padding = 0,
                 unsigned int dilation = 1);
  Array2D conv2d(Array2D& ifm,
                 Array2D& wgt,
                 unsigned int stride = 1,
                 unsigned int padding = 0,
                 unsigned int dilation = 1);
  Array3D conv3d(Array3D& ifm,
                 Array3D& wgt,
                 unsigned int stride = 1,
                 unsigned int padding = 0,
                 unsigned int dilation = 1);
};

Array1D Functional::conv1d(Array1D& ifm,
                           Array1D& wgt,
                           unsigned int stride,
                           unsigned int padding,
                           unsigned int dilation) {
  unsigned int width = (ifm.size() + 2 * padding - wgt.size()) / stride + 1;

  Array1D ofm(width, 0);

  for (int ofm_w = 0; ofm_w < ofm.size(); ofm_w++) {
    for (int wgt_w = 0; wgt_w < wgt.size(); wgt_w++) {
      int ifm_w = (ofm_w * stride - padding) + wgt_w * dilation;
      if (ifm_w >= 0 && ifm_w < ifm.size()) {
        ofm[ofm_w] += ifm[ifm_w] * wgt[wgt_w];
      }
    }
  }

  return ofm;
}

Array2D Functional::conv2d(Array2D& ifm,
                           Array2D& wgt,
                           unsigned int stride,
                           unsigned int padding,
                           unsigned int dilation) {
  unsigned int height = (ifm.size() + 2 * padding - wgt.size()) / stride + 1;
  unsigned int width = (ifm[0].size() + 2 * padding - wgt.size()) / stride + 1;

  Array2D ofm(height, Array1D(width, 0));

  for (int ofm_h = 0; ofm_h < ofm.size(); ofm_h++) {
    for (int ofm_w = 0; ofm_w < ofm[ofm_h].size(); ofm_w++) {
      for (int wgt_h = 0; wgt_h < wgt.size(); wgt_h++) {
        for (int wgt_w = 0; wgt_w < wgt.size(); wgt_w++) {
          int ifm_h = (ofm_h * stride - padding) + wgt_h * dilation;
          int ifm_w = (ofm_w * stride - padding) + wgt_w * dilation;
          if ((ifm_h >= 0 && ifm_h < ifm.size()) &&
              (ifm_w >= 0 && ifm_w < ifm[ofm_h].size())) {
            ofm[ofm_h][ofm_w] += ifm[ifm_h][ifm_w] * wgt[wgt_h][wgt_w];
          }
        }
      }
    }
  }

  return ofm;
}

Array3D Functional::conv3d(Array3D& ifm,
                           Array3D& wgt,
                           unsigned int stride,
                           unsigned int padding,
                           unsigned int dilation) {
  unsigned int depth = (ifm.size() + 2 * padding - wgt.size()) / stride + 1;
  unsigned int height = (ifm[0].size() + 2 * padding - wgt.size()) / stride + 1;
  unsigned int width =
      (ifm[0][0].size() + 2 * padding - wgt.size()) / stride + 1;

  Array3D ofm(depth, Array2D(height, Array1D(width, 0)));

  for (int ofm_b = 0; ofm_b < ofm.size(); ofm_b++) {
    for (int ofm_h = 0; ofm_h < ofm[ofm_b].size(); ofm_h++) {
      for (int ofm_w = 0; ofm_w < ofm[ofm_b][ofm_h].size(); ofm_w++) {
        for (int wgt_b = 0; wgt_b < wgt.size(); wgt_b++) {
          for (int wgt_h = 0; wgt_h < wgt.size(); wgt_h++) {
            for (int wgt_w = 0; wgt_w < wgt.size(); wgt_w++) {
              int ifm_b = (ofm_b * stride - padding) + wgt_b * dilation;
              int ifm_h = (ofm_h * stride - padding) + wgt_h * dilation;
              int ifm_w = (ofm_w * stride - padding) + wgt_w * dilation;
              if ((ifm_b >= 0 && ifm_b < ifm.size()) &&
                  (ifm_h >= 0 && ifm_h < ifm[ofm_b].size()) &&
                  (ifm_w >= 0 && ifm_w < ifm[ofm_b][ofm_h].size())) {
                ofm[ofm_b][ofm_h][ofm_w] +=
                    ifm[ifm_b][ifm_h][ifm_w] * wgt[wgt_b][wgt_h][wgt_w];
              }
            }
          }
        }
      }
    }
  }

  return ofm;
}

void print1d(const Array1D& data, const std::string& name) {
  std::cout << name << std::endl;
  for (int x = 0; x < data.size(); x++) {
    std::cout << data[x] << ", ";
  }
  std::cout << std::endl;
  std::cout << std::endl;
}

void print2d(const Array2D& data, const std::string& name) {
  std::cout << name << std::endl;
  for (int y = 0; y < data.size(); y++) {
    for (int x = 0; x < data[y].size(); x++) {
      std::cout << data[y][x] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void print3d(const Array3D& data, const std::string& name) {
  std::cout << name << std::endl;
  for (int z = 0; z < data.size(); z++) {
    for (int y = 0; y < data[z].size(); y++) {
      for (int x = 0; x < data[z][y].size(); x++) {
        std::cout << data[z][y][x] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void test() {
  Functional F;
  unsigned int ifm_size = 8;
  unsigned int wgt_size = 5;
  unsigned int stride = 2;
  unsigned int padding = 2;
  // unsigned int padding = wgt_size / 2; // same ifm/ofm size
  unsigned int dilation = 1;

  //// 1. 1D convolution
  std::cout << "\n--- 1D convolution ---\n" << std::endl;
  Array1D ifm1(ifm_size, 1);
  Array1D wgt1(wgt_size, 2);
  auto ofm1 = F.conv1d(ifm1, wgt1, stride, padding, dilation);
  print1d(ifm1, "input");
  print1d(wgt1, "kernel/weight/mask/filter");
  print1d(ofm1, "output");

  //// 2. 2D convolution
  std::cout << "\n--- 2D convolution ---\n" << std::endl;
  Array2D ifm2(ifm_size, Array1D(ifm_size, 1));
  Array2D wgt2(wgt_size, Array1D(wgt_size, 2));
  auto ofm2 = F.conv2d(ifm2, wgt2, stride, padding, dilation);
  print2d(ifm2, "input");
  print2d(wgt2, "kernel/weight/mask/filter");
  print2d(ofm2, "output");

  //// 3. 3D convolution
  std::cout << "\n--- 3D convolution ---\n" << std::endl;
  Array3D ifm3(ifm_size, Array2D(ifm_size, Array1D(ifm_size, 1)));
  Array3D wgt3(wgt_size, Array2D(wgt_size, Array1D(wgt_size, 2)));
  auto ofm3 = F.conv3d(ifm3, wgt3, stride, padding, dilation);
  print3d(ifm3, "input");
  print3d(wgt3, "kernel/weight/mask/filter");
  print3d(ofm3, "output");
}

int main(void) {
  test();
  return 0;
}
#else
#include <torch/extension.h>
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
  // const auto ifm_size = ifm_batch * ifm_channel * ifm_height * ifm_width;

  float *wgt_p = (float *)wgt.data_ptr();
  auto wgt_a = wgt.accessor<float, 4>();
  const auto wgt_batch = wgt_a.size(0);
  const auto wgt_channel = wgt_a.size(1);
  const auto wgt_height = wgt_a.size(2);
  const auto wgt_width = wgt_a.size(3);
  // const auto wgt_size = wgt_batch * wgt_channel * wgt_height * wgt_width;
  assert(ifm_channel == wgt_channel);

  const auto ofm_batch = ifm_batch * wgt_batch;
  const auto ofm_channel = 1;  // wgt_channel = ifm_channel;
  const auto ofm_height = (ifm_height + 2 * padding - wgt_height) / stride + 1;
  const auto ofm_width = (ifm_width + 2 * padding - wgt_width) / stride + 1;
  torch::Tensor ofm =
      torch::zeros({ofm_batch, ofm_channel, ofm_height, ofm_width});
  float *ofm_p = (float *)ofm.data_ptr();
  // const auto ofm_size = ofm_batch * ofm_channel * ofm_height * ofm_width;

  // for (int ofm_b = 0; ofm_b < ofm_batch; ofm_b++) {
  for (int ofm_h = 0; ofm_h < ofm_height; ofm_h++) {
    for (int ofm_w = 0; ofm_w < ofm_width; ofm_w++) {
      for (int wgt_c = 0; wgt_c < wgt_channel; wgt_c++) {
        for (int wgt_h = 0; wgt_h < wgt_height; wgt_h++) {
          for (int wgt_w = 0; wgt_w < wgt_width; wgt_w++) {
            int ifm_h = (ofm_h * stride - padding) + wgt_h * dilation;
            int ifm_w = (ofm_w * stride - padding) + wgt_w * dilation;
            int ifm_c = wgt_c;
            if ((ifm_h >= 0 && ifm_h < ifm_height) &&
                (ifm_w >= 0 && ifm_w < ifm_width)) {
              ofm_p[ofm_h * ofm_width + ofm_w] +=
                  ifm_p[ifm_c * ifm_height * ifm_width + ifm_h * ifm_width +
                        ifm_w] *
                  wgt_p[wgt_c * wgt_height * wgt_width + wgt_h * wgt_width +
                        wgt_w];
            }
          }
        }
      }
    }
  }
  // }

  return ofm;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2d", &conv2d, "naive conv2d with one cpu core");
}
#endif
