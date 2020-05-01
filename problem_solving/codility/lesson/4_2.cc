/**
 * File              : 4_1.cc
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2019.05.01
 * Last Modified Date: 2019.05.01
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */

/* check whether array A is a permutation.
 * https://app.codility.com/programmers/lessons/4-counting_elements/perm_check/
 */

#include <iostream>
#include <vector>

class Solution {
 public:
  int solution(int X, std::vector<int> &A) {
    int steps = X;
    // std::vector<bool> bitmap(A.size(), false);
    std::vector<bool> bitmap(A.size());  // default value is false
    // std::vector<bool> bitmap = {false}; // different with line 21

    for (unsigned int t = 0; t < A.size(); t++) {
      if (!bitmap[A[t]]) {
        bitmap[A[t]] = true;
        steps--;
        if (steps == 0) {
          return t;
        }
      }
    }
    return -1;
  }
};

int main(void) {
  Solution s;

  int ret;

  std::vector<int> a = {5, 1, 4, 2, 3, 6, 7, 9, 8};
  ret = s.solution(5, a);
  std::cout << "result: " << ret << std::endl;

  std::vector<int> b = {1, 3, 1, 4, 2, 3, 5, 4};
  ret = s.solution(5, b);
  std::cout << "result: " << ret << std::endl;

  return 0;
}
