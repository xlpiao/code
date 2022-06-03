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

class Solution {
 public:
  int solution(std::vector<int> &A) {
    int minimum = std::numeric_limits<int>::max();

    int left = std::accumulate(A.begin(), A.begin() + 1, 0);
    int right = std::accumulate(A.begin() + 1, A.end(), 0);
    int diff = std::abs(left - right);
    minimum = std::min(minimum, diff);
    // std::cout << "1: " << left << ", " << right << std::endl;

    for (unsigned int p = 2; p < A.size(); p++) {
      left += A[p - 1];
      right -= A[p - 1];
      diff = std::abs(left - right);
      minimum = std::min(minimum, diff);
      // std::cout << p << ": " << left << ", " << right << std::endl;
    }
    return minimum;
  }
};

int main(void) {
  Solution s;

  std::vector<int> a1{3, 1, 2, 4, 5};
  int ret = s.solution(a1);
  std::cout << ret << std::endl;

  return 0;
}
