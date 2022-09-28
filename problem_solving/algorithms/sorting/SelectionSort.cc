/**
 * File              : SelectionSort.cc
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
void selectionSort(std::vector<int> &arr) {
  std::cout << "\n" << __func__ << " output data: " << std::endl;
  int len = arr.size();

  for (int i = 0; i < len - 1; i++) {
    int minIdx = i;
    for (int j = i + 1; j < len; j++) {
      if (arr[minIdx] > arr[j]) {
        minIdx = j;
      }
    }
    int temp = arr[i];
    arr[i] = arr[minIdx];
    arr[minIdx] = temp;
    printArray(arr);
  }
}

int main(void) {
  std::vector<int> arr(10);

  initArray(arr);
  selectionSort(arr);

  return 0;
}
