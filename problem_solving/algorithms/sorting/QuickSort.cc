/**
 * File              : QuickSort.cc
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

//// Time Complexity: O(n*Log(n))
void quickSort(std::vector<int> &arr, int left, int right) {
  int pivot = arr[(left + right) / 2];
  std::cout << pivot << std::endl;
  int l = left;
  int r = right;

  while (l <= r) {
    while (arr[l] < pivot) {
      l++;
    }
    while (arr[r] > pivot) {
      r--;
    }
    if (l < r) {
      int temp = arr[l];
      arr[l] = arr[r];
      arr[r] = temp;
    }
    if (l <= r) {
      l++;
      r--;
    }
  }
  printf("left(%d), r(%d), l(%d), right(%d)\n", left, r, l, right);
  printArray(arr);

  if (left < r) {
    quickSort(arr, left, r);
  }
  if (l < right) {
    quickSort(arr, l, right);
  }
}

int main(void) {
  std::vector<int> arr(10);

  initArray(arr);
  std::cout << "\nquickSort output data: " << std::endl;
  quickSort(arr, 0, arr.size() - 1);

  return 0;
}
