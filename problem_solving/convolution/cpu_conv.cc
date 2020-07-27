/**
 * File              : cpu_conv.cc
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2020.06.15
 * Last Modified Date: 2020.06.15
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */

/* NOTE: convolutional output size */
/* output_size = (input_size + 2 * padding - kernel_size) / stride + 1 */
/* input_index = (output_index * stride - padding) + kernel_index * dilation */

#include <iostream>
#include <vector>

using Array1D = std::vector<float>;
using Array2D = std::vector<Array1D>;
using Array3D = std::vector<Array2D>;
using Array4D = std::vector<Array3D>;

inline auto Tensor(unsigned int d4, unsigned int d3, unsigned int d2, unsigned int d1, float value = 0.0f) {
  return Array4D(d4, Array3D(d3, Array2D(d2, Array1D(d1, value))));
}

class Functional {
public:
  Array1D conv1d(Array1D& ifm,
                 Array1D& wgt,
                 unsigned int stride = 1,
                 unsigned int padding = 0,
                 unsigned int dilation = 1);
  Array4D conv1d(Array4D& ifm,
                 Array4D& wgt,
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

Array4D Functional::conv1d(Array4D& ifm,
                           Array4D& wgt,
                           unsigned int stride,
                           unsigned int padding,
                           unsigned int dilation) {
  unsigned int width =
      (ifm[0][0][0].size() + 2 * padding - wgt[0][0][0].size()) / stride + 1;

  Array4D ofm{Tensor(1, 1, 1, width, 0.0f)};

  for (int ofm_x = 0; ofm_x < ofm[0][0][0].size(); ofm_x++) {
    for (int wgt_x = 0; wgt_x < wgt[0][0][0].size(); wgt_x++) {
      int ifm_x = (ofm_x * stride - padding) + wgt_x * dilation;
      if (ifm_x >= 0 && ifm_x < ifm[0][0][0].size()) {
        ofm[0][0][0][ofm_x] += ifm[0][0][0][ifm_x] * wgt[0][0][0][wgt_x];
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

void print(const Array4D& data, const std::string& name) {
  std::cout << name << std::endl;
  for (int z = 0; z < data.size(); z++) {
    for (int y = 0; y < data[z].size(); y++) {
      for (int x = 0; x < data[z][y].size(); x++) {
        for (int i = 0; i < data[0][0][0].size(); i++) {
          std::cout << data[0][0][0][i] << ", ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main(void) {
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

  Array4D ifm11{Tensor(1, 1, 1, ifm_size, 1)};
  Array4D wgt11{Tensor(1, 1, 1, wgt_size, 2)};
  auto ofm11 = F.conv1d(ifm11, wgt11, stride, padding, dilation);
  print(ifm11, "input");
  print(wgt11, "kernel/weight/mask/filter");
  print(ofm11, "output");

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

  return 0;
}
