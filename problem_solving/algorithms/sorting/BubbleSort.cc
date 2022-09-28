/**
 * File              : BubbleSort.cc
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2021.05.23
 * Last Modified Date: 2022.09.27
 * Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>
 */
#include <cstdlib>
#include <iostream>
#include <vector>

void printArray(std::vector<int> &arr) {
  int len = arr.size();

  for (int i = 0; i < len; i++) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}

void initArray(std::vector<int> &arr) {
  int len = arr.size();

  for (int i = 0; i < len; i++) {
    std::srand(i);
    arr[i] = std::rand() % 10;
  }
  std::cout << "\ninput data: " << std::endl;
  printArray(arr);
}

//// Time Complexity: O(n^2)
void bubbleSort(std::vector<int> &arr) {
  std::cout << "\n" << __func__ << " output data: " << std::endl;
  int len = arr.size();

  for (int i = 1; i < len; i++) {
    for (int j = 0; j < len - i; j++) {
      if (arr[j] > arr[j + 1]) {
        int temp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = temp;
      }
    }
    printArray(arr);
  }
}

int main(void) {
  std::vector<int> arr(10);

  initArray(arr);
  bubbleSort(arr);

  return 0;
}
