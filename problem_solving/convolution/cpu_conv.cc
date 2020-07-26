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

class Functional {
public:
  Array1D conv1d(Array1D& input,
                 Array1D& kernel,
                 unsigned int stride = 1,
                 unsigned int padding = 0,
                 unsigned int dilation = 1);
  Array2D conv2d(Array2D& input,
                 Array2D& kernel,
                 unsigned int stride = 1,
                 unsigned int padding = 0,
                 unsigned int dilation = 1);
  Array3D conv3d(Array3D& input,
                 Array3D& kernel,
                 unsigned int stride = 1,
                 unsigned int padding = 0,
                 unsigned int dilation = 1);
};

Array1D Functional::conv1d(Array1D& input,
                           Array1D& kernel,
                           unsigned int stride,
                           unsigned int padding,
                           unsigned int dilation) {
  unsigned int size = (input.size() + 2 * padding - kernel.size()) / stride + 1;

  Array1D output(size, 0);

  for (int i = 0; i < output.size(); i += 1) {
    for (int k = 0; k < kernel.size(); k++) {
      int col = (i * stride - padding) + k * dilation;
      if (col >= 0 && col < input.size()) {
        output[i] += input[col] * kernel[k];
      }
    }
  }

  return output;
}

Array2D Functional::conv2d(Array2D& input,
                           Array2D& kernel,
                           unsigned int stride,
                           unsigned int padding,
                           unsigned int dilation) {
  unsigned int row_size =
      (input.size() + 2 * padding - kernel.size()) / stride + 1;
  unsigned int col_size =
      (input[0].size() + 2 * padding - kernel.size()) / stride + 1;

  Array2D output(row_size, Array1D(col_size, 0));

  for (int i = 0; i < output.size(); i++) {
    for (int j = 0; j < output[i].size(); j++) {
      for (int m = 0; m < kernel.size(); m++) {
        for (int n = 0; n < kernel.size(); n++) {
          int row = (i * stride - padding) + m * dilation;
          int col = (j * stride - padding) + n * dilation;
          if ((row >= 0 && row < input.size()) &&
              (col >= 0 && col < input[i].size())) {
            output[i][j] += input[row][col] * kernel[m][n];
          }
        }
      }
    }
  }

  return output;
}

Array3D Functional::conv3d(Array3D& input,
                           Array3D& kernel,
                           unsigned int stride,
                           unsigned int padding,
                           unsigned int dilation) {
  unsigned int row_size =
      (input.size() + 2 * padding - kernel.size()) / stride + 1;
  unsigned int col_size =
      (input[0].size() + 2 * padding - kernel.size()) / stride + 1;
  unsigned int depth_size =
      (input[0][0].size() + 2 * padding - kernel.size()) / stride + 1;

  Array3D output(row_size, Array2D(col_size, Array1D(depth_size, 0)));

  for (int i = 0; i < output.size(); i++) {
    for (int j = 0; j < output[i].size(); j++) {
      for (int k = 0; k < output[i][j].size(); k++) {
        for (int m = 0; m < kernel.size(); m++) {
          for (int n = 0; n < kernel.size(); n++) {
            for (int l = 0; l < kernel.size(); l++) {
              int row = (i * stride - padding) + m * dilation;
              int col = (j * stride - padding) + n * dilation;
              int depth = (k * stride - padding) + l * dilation;
              if ((row >= 0 && row < input.size()) &&
                  (col >= 0 && col < input[i].size()) &&
                  (depth >= 0 && depth < input[i][j].size())) {
                output[i][j][k] += input[row][col][depth] * kernel[m][n][l];
              }
            }
          }
        }
      }
    }
  }

  return output;
}

void print1d(const Array1D& data, const std::string& name) {
  std::cout << name << std::endl;
  for (auto it : data) {
    std::cout << it << ", ";
  }
  std::cout << std::endl;
  std::cout << std::endl;
}

void print2d(const Array2D& data, const std::string& name) {
  std::cout << name << std::endl;
  for (int i = 0; i < data.size(); i++) {
    for (int j = 0; j < data[i].size(); j++) {
      std::cout << data[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void print3d(const Array3D& data, const std::string& name) {
  std::cout << name << std::endl;
  for (int i = 0; i < data.size(); i++) {
    for (int j = 0; j < data[i].size(); j++) {
      for (int k = 0; k < data[i][j].size(); k++) {
        std::cout << data[i][j][k] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main(void) {
  Functional F;
  unsigned int input_size = 8;
  unsigned int kernel_size = 5;
  unsigned int stride = 2;
  unsigned int padding = 2;
  // unsigned int padding = kernel_size / 2; // same input/output size
  unsigned int dilation = 1;

  //// 1. 1D convolution
  std::cout << "\n--- 1D convolution ---\n" << std::endl;
  Array1D input1(input_size, 1);
  Array1D kernel1(kernel_size, 2);
  auto output1 = F.conv1d(input1, kernel1, stride, padding, dilation);
  print1d(input1, "input");
  print1d(kernel1, "kernel");
  print1d(output1, "output");

  //// 2. 2D convolution
  std::cout << "\n--- 2D convolution ---\n" << std::endl;
  Array2D input2(input_size, Array1D(input_size, 1));
  Array2D kernel2(kernel_size, Array1D(kernel_size, 2));
  auto output2 = F.conv2d(input2, kernel2, stride, padding, dilation);
  print2d(input2, "input");
  print2d(kernel2, "kernel");
  print2d(output2, "output");

  //// 3. 3D convolution
  std::cout << "\n--- 3D convolution ---\n" << std::endl;
  Array3D input3(input_size, Array2D(input_size, Array1D(input_size, 1)));
  Array3D kernel3(kernel_size, Array2D(kernel_size, Array1D(kernel_size, 2)));
  auto output3 = F.conv3d(input3, kernel3, stride, padding, dilation);
  print3d(input3, "input");
  print3d(kernel3, "kernel");
  print3d(output3, "output");

  return 0;
}
