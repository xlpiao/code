#include <iostream>
#include <vector>

using Array1D = std::vector<float>;
using Array2D = std::vector<Array1D>;

class Functional {
 public:
  Array1D conv1d(Array1D& input, Array1D& filter, int stride = 1,
                 int padding = 0);
  Array2D conv2d(Array2D& input, Array2D& filter, int stride = 1,
                 int padding = 0);
};

Array1D Functional::conv1d(Array1D& input, Array1D& filter, int stride,
                           int padding) {
  const unsigned int size = input.size();
  Array1D output(size, 0);

  for (int i = 0; i < input.size(); i++) {
    for (int j = 0; j < filter.size(); j++) {
      output[i] += input[i] * filter[j];
    }
  }

  return output;
}

Array2D Functional::conv2d(Array2D& input, Array2D& filter, int stride,
                           int padding) {
  const unsigned int row = input.size();
  const unsigned int col = input[0].size();
  Array2D output(row, Array1D(col, 0));

  for (int i = 0; i < input.size(); i++) {
    for (int j = 0; j < input[i].size(); j++) {
      for (int m = 0; m < filter.size(); m++) {
        for (int n = 0; n < filter.size(); n++) {
          output[i][j] += input[i][j] * filter[m][n];
        }
      }
    }
  }

  return output;
}

int main(void) {
  Functional F;

  //// 1. 1D array convolution
  Array1D input1(5, 1);
  Array1D filter1(3, 2);

  auto output1 = F.conv1d(input1, filter1);
  for (auto it : output1) {
    std::cout << it << ", ";
  }
  std::cout << std::endl;
  std::cout << std::endl;

  //// 2. 2D array convolution
  Array2D input2(5, Array1D(5, 1));
  Array2D filter2(3, Array1D(3, 2));

  auto output2 = F.conv2d(input2, filter2);
  for (int i = 0; i < output2.size(); i++) {
    for (int j = 0; j < output2[i].size(); j++) {
      std::cout << output2[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
