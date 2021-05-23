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
    arr[i] = 10 + std::rand() / ((RAND_MAX + 1u) / 6);
  }
  printArray(arr);
}

//// Time Complexity: O(n^2)
void bubbleSort(std::vector<int> &arr) {
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

//// Time Complexity: O(n^2)
void selectionSort(std::vector<int> &arr) {
  int len = arr.size();

  for (int i = 0; i < len - 1; i++) {
    int minIdx = i;
    for (int j = i + 1; j < len; j++) {
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
  int len = arr.size();

  for (int i = 0; i < len - 1; i++) {
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
    if (i < j) {
      int temp = arr[i];
      arr[i] = arr[j];
      arr[j] = temp;
    }
    if (i <= j) {
      i++;
      j--;
    }
  }
  printArray(arr);

  if (left < j) {
    quickSort(arr, left, j);
  }
  if (i < right) {
    quickSort(arr, i, right);
  }
}

//// Time Complexity: O(n*Log(n))
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

  int i = 0;
  int j = 0;
  int k = left;
  while (i < leftSize && j < rightSize) {
    if (leftArr[i] < rightArr[j]) {
      arr[k] = leftArr[i];
      i++;
    } else {
      arr[k] = rightArr[j];
      j++;
    }
    k++;
  }
  while (i < leftSize) {
    arr[k] = leftArr[i];
    i++;
    k++;
  }
  while (j < rightSize) {
    arr[k] = rightArr[j];
    j++;
    k++;
  }
}
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
  quickSort(arr, 0, arr.size() - 1);

  std::cout << "\nOrigin Array: " << std::endl;
  initArray(arr);
  std::cout << "\nMerge Sorting ..." << std::endl;
  mergeSort(arr, 0, arr.size() - 1);

  return 0;
}
