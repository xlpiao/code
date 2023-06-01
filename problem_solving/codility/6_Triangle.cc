
// you can use includes, for example:
// #include <algorithm>

// you can write to stdout for debugging purposes, e.g.
// cout << "this is a debug message" << endl;

#include <algorithm>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

int solution(vector<int> &A) {
  const int n = A.size();
  if (n < 3) return 0;
  std::sort(A.begin(), A.end());

  for (auto i = 0; i < n - 2; ++i) {
    if (A[i] > A[i + 2] - A[i + 1]) {
      return 1;
    }
  }
  return 0;
}

int main(void) {
  vector<int> A{10, 2, 5, 1, 8, 20};
  auto ans = solution(A);
  cout << "\n" << ans << "\n" << endl;

  return 0;
}
