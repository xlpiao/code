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

  for (int ofm_x = 0; ofm_x < ofm.size(); ofm_x++) {
    for (int wgt_x = 0; wgt_x < wgt.size(); wgt_x++) {
      int ifm_x = (ofm_x * stride - padding) + wgt_x * dilation;
      if (ifm_x >= 0 && ifm_x < ifm.size()) {
        ofm[ofm_x] += ifm[ifm_x] * wgt[wgt_x];
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

  for (int ofm_y = 0; ofm_y < ofm.size(); ofm_y++) {
    for (int ofm_x = 0; ofm_x < ofm[ofm_y].size(); ofm_x++) {
      for (int wgt_y = 0; wgt_y < wgt.size(); wgt_y++) {
        for (int wgt_x = 0; wgt_x < wgt.size(); wgt_x++) {
          int ifm_y = (ofm_y * stride - padding) + wgt_y * dilation;
          int ifm_x = (ofm_x * stride - padding) + wgt_x * dilation;
          if ((ifm_y >= 0 && ifm_y < ifm.size()) &&
              (ifm_x >= 0 && ifm_x < ifm[ofm_y].size())) {
            ofm[ofm_y][ofm_x] += ifm[ifm_y][ifm_x] * wgt[wgt_y][wgt_x];
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

  for (int ofm_z = 0; ofm_z < ofm.size(); ofm_z++) {
    for (int ofm_y = 0; ofm_y < ofm[ofm_z].size(); ofm_y++) {
      for (int ofm_x = 0; ofm_x < ofm[ofm_z][ofm_y].size(); ofm_x++) {
        for (int wgt_z = 0; wgt_z < wgt.size(); wgt_z++) {
          for (int wgt_y = 0; wgt_y < wgt.size(); wgt_y++) {
            for (int wgt_x = 0; wgt_x < wgt.size(); wgt_x++) {
              int ifm_z = (ofm_z * stride - padding) + wgt_z * dilation;
              int ifm_y = (ofm_y * stride - padding) + wgt_y * dilation;
              int ifm_x = (ofm_x * stride - padding) + wgt_x * dilation;
              if ((ifm_z >= 0 && ifm_z < ifm.size()) &&
                  (ifm_y >= 0 && ifm_y < ifm[ofm_z].size()) &&
                  (ifm_x >= 0 && ifm_x < ifm[ofm_z][ofm_y].size())) {
                ofm[ofm_z][ofm_y][ofm_x] +=
                    ifm[ifm_z][ifm_y][ifm_x] * wgt[wgt_z][wgt_y][wgt_x];
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
  const auto ifm_b = ifm_a.size(0);
  const auto ifm_c = ifm_a.size(1);
  const auto ifm_h = ifm_a.size(2);
  const auto ifm_w = ifm_a.size(3);
  // const auto ifm_size = ifm_b * ifm_c * ifm_h * ifm_w;

  float *wgt_p = (float *)wgt.data_ptr();
  auto wgt_a = wgt.accessor<float, 4>();
  const auto wgt_b = wgt_a.size(0);
  const auto wgt_c = wgt_a.size(1);
  const auto wgt_h = wgt_a.size(2);
  const auto wgt_w = wgt_a.size(3);
  // const auto wgt_size = wgt_b * wgt_c * wgt_h * wgt_w;

  const auto ofm_b = ifm_b;
  const auto ofm_c = wgt_c;
  const auto ofm_h = (ifm_h + 2 * padding - wgt_h) / stride + 1;
  const auto ofm_w = (ifm_w + 2 * padding - wgt_w) / stride + 1;
  torch::Tensor ofm = torch::zeros({ofm_b, ofm_c, ofm_h, ofm_w});
  float *ofm_p = (float *)ofm.data_ptr();
  // auto ofm_a = ofm.accessor<float, 4>();
  // const auto ofm_size = ofm_b * ofm_c * ofm_h * ofm_w;

  for (int ofm_y = 0; ofm_y < ofm_h; ofm_y++) {
    for (int ofm_x = 0; ofm_x < ofm_w; ofm_x++) {
      for (int wgt_y = 0; wgt_y < wgt_h; wgt_y++) {
        for (int wgt_x = 0; wgt_x < wgt_w; wgt_x++) {
          int ifm_y = (ofm_y * stride - padding) + wgt_y * dilation;
          int ifm_x = (ofm_x * stride - padding) + wgt_x * dilation;
          if ((ifm_y >= 0 && ifm_y < ifm_h) && (ifm_x >= 0 && ifm_x < ifm_w)) {
            ofm_p[ofm_y * ofm_w + ofm_x] +=
                ifm_p[ifm_y * ifm_w + ifm_x] * wgt_p[wgt_y * wgt_w + wgt_x];
          }
        }
      }
    }
  }

  return ofm;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2d", &conv2d, "naive conv2d with one cpu core");
}
#endif
