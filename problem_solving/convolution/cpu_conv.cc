/**
 * File              : cpu_conv.cc
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2020.06.15
 * Last Modified Date: 2020.06.15
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */

/* NOTE: convolutional output size */
/* output_size = (intput_size + 2 * padding - kernel_size) / stride + 1 */
/* input_index = output_index * stride - padding + kernel_index */

#include <iostream>
#include <vector>

using Array1D = std::vector<float>;
using Array2D = std::vector<Array1D>;

class Functional {
 public:
  Array1D conv1d(Array1D& input, Array1D& filter, const unsigned int stride = 1,
                 const unsigned int padding = 0);
  Array2D conv2d(Array2D& input, Array2D& filter, const unsigned int stride = 1,
                 const unsigned int padding = 0);
};

Array1D Functional::conv1d(Array1D& input, Array1D& filter,
                           const unsigned int stride,
                           const unsigned int padding) {
  const unsigned int size =
      (input.size() + 2 * padding - filter.size()) / stride + 1;
  Array1D output(size, 0);

  for (int i = 0; i < output.size(); i += 1) {
    for (int k = 0; k < filter.size(); k++) {
      int offset = i * stride - padding + k;
      if (offset >= 0 && offset < input.size()) {
        output[i] += input[offset] * filter[k];
      }
    }
  }

  return output;
}

Array2D Functional::conv2d(Array2D& input, Array2D& filter,
                           const unsigned int stride,
                           const unsigned int padding) {
  const unsigned int row_size =
      (input.size() + 2 * padding - filter.size()) / stride + 1;
  const unsigned int col_size =
      (input[0].size() + 2 * padding - filter.size()) / stride + 1;
  Array2D output(row_size, Array1D(col_size, 0));

  for (int i = 0; i < output.size(); i++) {
    for (int j = 0; j < output[i].size(); j++) {
      for (int m = 0; m < filter.size(); m++) {
        for (int n = 0; n < filter.size(); n++) {
          int row = (i * stride - padding) + m;
          int col = (j * stride - padding) + n;
          if (row >= 0 && row < input.size() && col >= 0 &&
              col < input[i].size()) {
            output[i][j] += input[row][col] * filter[m][n];
          }
        }
      }
    }
  }

  return output;
}

int main(void) {
  Functional F;
  const unsigned int input_size = 8;
  const unsigned int filter_size = 5;
  const unsigned int stride = 2;
  const unsigned int padding = 2;

  //// 1. 1D array convolution
  std::cout << "\n--- 1D convolution ---\n" << std::endl;
  Array1D input1(input_size, 1);
  Array1D filter1(filter_size, 2);

  std::cout << "input: " << std::endl;
  for (auto it : input1) {
    std::cout << it << ", ";
  }
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "filter: " << std::endl;
  for (auto it : filter1) {
    std::cout << it << ", ";
  }
  std::cout << std::endl;
  std::cout << std::endl;

  auto output1 = F.conv1d(input1, filter1, stride, padding);
  std::cout << "output: " << std::endl;
  for (auto it : output1) {
    std::cout << it << ", ";
  }
  std::cout << std::endl;
  std::cout << std::endl;

  //// 2. 2D array convolution
  std::cout << "\n--- 2D convolution ---\n" << std::endl;
  Array2D input2(input_size, Array1D(input_size, 1));
  Array2D filter2(filter_size, Array1D(filter_size, 2));

  std::cout << "input: " << std::endl;
  for (int i = 0; i < input2.size(); i++) {
    for (int j = 0; j < input2[i].size(); j++) {
      std::cout << input2[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "filter: " << std::endl;
  for (int i = 0; i < filter2.size(); i++) {
    for (int j = 0; j < filter2[i].size(); j++) {
      std::cout << filter2[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto output2 = F.conv2d(input2, filter2, stride, padding);
  for (int i = 0; i < output2.size(); i++) {
    for (int j = 0; j < output2[i].size(); j++) {
      std::cout << output2[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
