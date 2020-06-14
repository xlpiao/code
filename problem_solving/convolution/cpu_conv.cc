#include <iostream>
#include <vector>

using Array1D = std::vector<float>;

class Functional {
 public:
  Array1D conv1d(Array1D& input, Array1D& filter, int stride = 1,
                 int padding = 1);
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

int main(void) {
  Array1D input(5, 1);
  Array1D filter(3, 2);

  Functional F;
  auto output = F.conv1d(input, filter);
  for (auto it : output) {
    std::cout << it << std::endl;
  }

  return 0;
}
