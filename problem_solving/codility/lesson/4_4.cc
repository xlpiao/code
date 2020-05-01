/**
 * File              : 4_4.cc
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2019.05.01
 * Last Modified Date: 2019.05.06
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */

/* Find the smallest positive integer that does not occur in a given sequence.
 * https://app.codility.com/programmers/lessons/4-counting_elements/missing_integer/
 */

#include <iostream>
#include <vector>

class Solution {
 public:
  int solution(std::vector<int> &A) {
    int N = 100000;
    std::vector<int> bitmap(N);
    for (unsigned int i = 0; i < A.size(); i++) {
      if (A[i] >= 1 && A[i] <= N) {
        bitmap[A[i] - 1] = A[i];
      }
    }
    for (int i = 0; i < N; i++) {
      if (bitmap[i] == 0) {
        return (i + 1);
      }
    }
    return N + 1;
  }
};

int main(void) {
  Solution s;

  std::vector<int> a;
  for (int i = 0; i < 100000; i++) {
    a.push_back(i + 1);
  }
  int ret = s.solution(a);
  std::cout << "result: " << ret << std::endl;

  return 0;
}
