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

#include <algorithm>  // std::sort
#include <iostream>
#include <vector>

class Solution {
 public:
  int solution(std::vector<int> &A) {
    std::sort(A.begin(), A.end());
    for (unsigned int i = 0; i < A.size(); i++) {
      // if (A[i] != (i + 1)) { // same with follows
      if (A[i] ^ (i + 1)) {  // 1^1=0, 0^0=0, 1^0=1
        return 0;
      }
    }
    return 1;
  }
};

int main(void) {
  Solution s;

  std::vector<int> a = {5, 7, 4, 2, 8, 6, 1, 9, 9, 3};
  int ret = s.solution(a);
  std::cout << "result: " << ret << std::endl;

  return 0;
}
