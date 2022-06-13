#include <iostream>
#include <vector>

using std::cout;
using std::endl;

template <typename T>
std::vector<T> quicksort(const std::vector<T>& arr) {
  // base case, arrays with 0 or 1 element are already "sorted"
  if (arr.size() < 2) return arr;

  // recursive case
  const T* pivot = &arr.front() + arr.size() / 2 - 1;
  std::vector<T> smaller;
  std::vector<T> larger;

  for (const T* item = &arr.front(); item <= &arr.back(); item++) {
    if (item == pivot) continue;
    if (*item <= *pivot)
      smaller.push_back(*item);
    else
      larger.push_back(*item);
  }

  std::vector<T> sorted_less = quicksort(smaller);
  std::vector<T> sorted_greater = quicksort(larger);
  // concatenate smaller part, pivot and larger part
  sorted_less.push_back(*pivot);
  sorted_less.insert(
      sorted_less.end(), sorted_greater.begin(), sorted_greater.end());

  return sorted_less;
}

int main() {
  std::vector<int> arr = {69, 60, 38, 82, 99, 15, 8,  94, 30, 42, 35, 40,
                          63, 1,  49, 66, 93, 83, 20, 32, 87, 6,  78, 17,
                          2,  61, 91, 25, 7,  4,  97, 31, 23, 67, 95, 47,
                          55, 92, 37, 59, 73, 81, 74, 41, 39};
  cout << arr.size() << endl;
  std::vector<int> sorted = quicksort(arr);
  for (int num : sorted) {
    cout << num << " ";
  }
  cout << endl;
}
