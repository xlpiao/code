/**
 * File              : 2_1.cc
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2018.08.11
 * Last Modified Date: 2019.04.29
 * Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>
 */

/* Find value that occurs in odd number of elements.
 * https://app.codility.com/programmers/lessons/2-arrays/odd_occurrences_in_array/
 */

#include <iostream>
#include <vector>

class Solution {
 public:
  int solution(vector<int> &A) {
    int n = A.size();

    for (int i = 1; i < n; ++i) {
      A[i] = A[i] ^ A[i - 1];
    }

    return A[n - 1];
  }
};

int main(void) {
  Solution s;

  std::vector<int> A{9, 3, 9, 3, 9, 7, 9};

  int result = s.solution(A);
  std::cout << "Result is " << result << std::endl;

  return 0;
}
