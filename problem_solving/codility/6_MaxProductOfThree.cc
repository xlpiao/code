
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
  int n = A.size();
  sort(A.begin(), A.end());

  int prod1 = A[n - 1] * A[n - 2] * A[n - 3];
  int prod2 = A[0] * A[1] * A[n - 1];
  return max(prod1, prod2);
}

int main(void) {
  vector<int> A{-3, 1, 2, -2, 5, 6};
  auto ans = solution(A);
  cout << "\n" << ans << "\n" << endl;

  return 0;
}
