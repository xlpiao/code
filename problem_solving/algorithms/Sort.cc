#include <cstdlib>
#include <iostream>
#include <vector>

void printArray(std::vector<int> &arr) {
  int length = arr.size();

  for (int i = 0; i < length; i++) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}

void initArray(std::vector<int> &arr) {
  int length = arr.size();

  for (int i = 0; i < length; i++) {
    arr[i] = 10 + std::rand() / ((RAND_MAX + 1u) / 6);
  }
  printArray(arr);
}

//// Time Complexity: O(n^2)
void bubbleSort(std::vector<int> &arr) {
  int length = arr.size();

  for (int i = 1; i < length; i++) {
    for (int j = 0; j < length - i; j++) {
      if (arr[j] > arr[j + 1]) {
        int temp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = temp;
      }
    }
    printArray(arr);
  }
}

//// Time Complexity: O(n^2)
void selectionSort(std::vector<int> &arr) {
  int length = arr.size();

  for (int i = 0; i < length - 1; i++) {
    int minIdx = i;
    for (int j = i + 1; j < length; j++) {
      if (arr[j] < arr[minIdx]) {
        minIdx = j;
      }
    }
    int temp = arr[i];
    arr[i] = arr[minIdx];
    arr[minIdx] = temp;
    printArray(arr);
  }
}

//// Time Complexity: O(n^2)
void insertionSort(std::vector<int> &arr) {
  int length = arr.size();

  for (int i = 0; i < length - 1; i++) {
    for (int j = i + 1; j > 0; j--) {
      if (arr[j] < arr[j - 1]) {
        int temp = arr[j];
        arr[j] = arr[j - 1];
        arr[j - 1] = temp;
      }
    }
    printArray(arr);
  }
}

//// Time Complexity: O(n*Log(n))
void quickSort(std::vector<int> &arr, int left, int right) {
  int pivot = arr[(left + right) / 2];
  int i = left;
  int j = right;

  while (i <= j) {
    while (arr[i] < pivot) {
      i++;
    }
    while (arr[j] > pivot) {
      j--;
    }
    if (i <= j) {
      int temp = arr[i];
      arr[i] = arr[j];
      arr[j] = temp;
      i++;
      j--;
    }
  }

  if (left < j) {
    quickSort(arr, left, j);
  }
  if (i < right) {
    quickSort(arr, i, right);
  }
  printArray(arr);
}

int main(void) {
  std::vector<int> arr(10);

  std::cout << "\nOrigin Array: " << std::endl;
  initArray(arr);
  std::cout << "\nBubble Sorting ..." << std::endl;
  bubbleSort(arr);

  std::cout << "\nOrigin Array: " << std::endl;
  initArray(arr);
  std::cout << "\nSelection Sorting ..." << std::endl;
  selectionSort(arr);

  std::cout << "\nOrigin Array: " << std::endl;
  initArray(arr);
  std::cout << "\nInsertion Sorting ..." << std::endl;
  insertionSort(arr);

  std::cout << "\nOrigin Array: " << std::endl;
  initArray(arr);
  std::cout << "\nQuick Sorting ..." << std::endl;
  quickSort(arr, 0, arr.size());

  return 0;
}
