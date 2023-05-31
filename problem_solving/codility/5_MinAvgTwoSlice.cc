
// you can use includes, for example:
// #include <algorithm>

// you can write to stdout for debugging purposes, e.g.
// cout << "this is a debug message" << endl;

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

int solution(vector<int> &A) {
  size_t n = A.size();
  vector<int> prefix_sum(n + 1, 0);
  int sum = 0;

  for (size_t i = 0; i < n; i++) {
    sum += A[i];
    prefix_sum[i + 1] = sum;
  }

  float min_avg = std::numeric_limits<float>::max();
  int min_idx = n;
  for (size_t i = 0; i < n; i++) {
    size_t max_len = min(3, static_cast<int>(n - i));
    for (size_t j = 2; j <= max_len; j++) {
      float avg = static_cast<float>((prefix_sum[i + j] - prefix_sum[i])) / j;
      if (avg < min_avg) {
        min_idx = i;
        min_avg = avg;
      }
    }
  }
  return min_idx;
}

int main(void) {
  vector<int> A{4, 2, 2, 5, 1, 5, 8};
  auto ans = solution(A);
  cout << ans << endl;

  return 0;
}
