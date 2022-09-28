/**
 * File              : MergeSort.cc
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

void merge(std::vector<int> &arr, int left, int mid, int right) {
  int leftSize = mid - left + 1;
  int rightSize = right - (mid + 1) + 1;

  int leftArr[leftSize];
  int rightArr[rightSize];
  for (int i = 0; i < leftSize; i++) {
    leftArr[i] = arr[i + left];
  }
  for (int i = 0; i < rightSize; i++) {
    rightArr[i] = arr[i + mid + 1];
  }

  int l = 0;
  int r = 0;
  int idx = left;
  while (l < leftSize && r < rightSize) {
    if (leftArr[l] < rightArr[r]) {
      arr[idx] = leftArr[l];
      l++;
    } else {
      arr[idx] = rightArr[r];
      r++;
    }
    idx++;
  }
  while (l < leftSize) {
    arr[idx] = leftArr[l];
    l++;
    idx++;
  }
  while (r < rightSize) {
    arr[idx] = rightArr[r];
    r++;
    idx++;
  }
}

//// Time Complexity: O(n*Log(n))
void mergeSort(std::vector<int> &arr, int left, int right) {
  if (left < right) {
    int mid = (left + right) / 2;
    mergeSort(arr, left, mid);
    mergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
  }
  printArray(arr);
}

int main(void) {
  std::vector<int> arr(10);

  initArray(arr);
  std::cout << "\nmergeSort output data: " << std::endl;
  mergeSort(arr, 0, arr.size() - 1);

  return 0;
}
