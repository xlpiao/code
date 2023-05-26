/**
 * File              : 3_3.cc
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2019.05.01
 * Last Modified Date: 2019.05.01
 * Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>
 */

/* Minimize the value |(A[0] + ... + A[P-1]) - (A[P] + ... + A[N-1])|.
 * https://app.codility.com/programmers/lessons/3-time_complexity/tape_equilibrium/
 */

#include <cmath>  // std::abs
#include <iostream>
#include <limits>   // std::numeric_limits
#include <numeric>  // std::accumulate
#include <vector>

using namespace std;

class Solution {
 public:
  int solution(std::vector<int> &A) {
    int n = A.size();

    long sum = 0;
    for (int i = 0; i < n; ++i) {
      sum += A[i];
    }

    int minDiff = std::numeric_limits<int>::max();
    long left = 0;
    long right = 0;

    for (int p = 0; p < n - 1; p++) {
      left += A[p];
      right = sum - left;
      int diff = (int)abs(left - right);
      minDiff = min(diff, minDiff);
    }
    return minDiff;
  }
};

int main(void) {
  Solution s;

  std::vector<int> a1{3, 1, 2, 4, 5};
  int ret = s.solution(a1);
  std::cout << ret << std::endl;

  return 0;
}
