/**
 * File              : 3_2.cc
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2019.04.29
 * Last Modified Date: 2019.04.29
 * Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>
 */

/* Find the missing element in a given permutation.
 * https://app.codility.com/programmers/lessons/3-time_complexity/perm_missing_elem/
 */

#include <iostream>
#include <numeric>
#include <vector>

class Solution {
 public:
  int solution(std::vector<int> &A) {
    /* TODO */
    unsigned long max_num = A.size() + 1;
    unsigned long sum = (max_num * (1 + max_num)) / 2;
    /* NOTE: sum from 1 ~ max_num, n*(A1+An)/2 */
    int missing = sum - std::accumulate(A.begin(), A.end(), 0);
    return missing;
  }
};

int main(void) {
  Solution s;

  std::vector<int> A{1, 5, 4, 3};
  std::vector<int> B{2, 4, 3};
  std::vector<int> C{1, 2, 5, 4, 6};
  std::vector<int> D;

  int missing = s.solution(A);
  std::cout << missing << std::endl;
  missing = s.solution(B);
  std::cout << missing << std::endl;
  missing = s.solution(C);
  std::cout << missing << std::endl;
  missing = s.solution(D);
  std::cout << missing << std::endl;

  return 0;
}
