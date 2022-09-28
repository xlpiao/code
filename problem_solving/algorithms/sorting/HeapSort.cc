/**
 * File              : HeapSort.cc
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2021.05.23
 * Last Modified Date: 2021.05.23
 * Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>
 */
#include <iostream>
#include <vector>

using Tree = std::vector<int>;

void printArray(Tree& arr) {
  int len = arr.size();

  for (int i = 0; i < len; ++i) {
    std::cout << arr[i] << " ";
  }
  std::cout << std::endl;
}

void initArray(Tree& arr) {
  int len = arr.size();

  for (int i = 0; i < len; i++) {
    std::srand(i);
    arr[i] = std::rand() % 10;
  }
  std::cout << "\ninput data: " << std::endl;
  printArray(arr);
}

//// heapify a branch (parent, left, right)
void heapify(Tree& arr, int len, int i) {
  if (i >= len) {
    return;
  }

  int left = 2 * i + 1;
  int right = 2 * i + 2;
  int largest = i;

  if (left < len && arr[left] > arr[largest]) {
    largest = left;
  }

  if (right < len && arr[right] > arr[largest]) {
    largest = right;
  }

  if (largest != i) {
    std::cout << "arr[" << i << "] = " << arr[i] << ", "
              << "arr[" << largest << "] = " << arr[largest] << std::endl;
    std::swap(arr[i], arr[largest]);
    heapify(arr, len, largest);
  }
}

//// heapify each layer: from last second layer to top layer
//// Time Complexity: O(n*Log(n))
void buildHeap(Tree& arr) {
  int len = arr.size();
  int last = len - 1;
  int parent = (last - 1) / 2;

  for (int i = parent; i >= 0; i--) {
    heapify(arr, len, i);
  }
}

void heapSort(Tree& arr) {
  std::cout << "\nBuild Heap Array: ";
  buildHeap(arr);
  printArray(arr);

  std::cout << "\nHeap Sorting...:" << std::endl;
  int len = arr.size();
  for (int i = len - 1; i >= 0; i--) {
    std::cout << "before: ";
    printArray(arr);
    std::swap(arr[0], arr[i]);
    std::cout << "swap: ";
    printArray(arr);
    heapify(arr, i, 0);
    std::cout << "after: ";
    printArray(arr);
  }
}

int main(void) {
  Tree arr(10);
  initArray(arr);

  std::cout << "\nOriginal Array: ";
  printArray(arr);

  heapSort(arr);

  std::cout << "\nSorted Array: ";
  printArray(arr);
}
