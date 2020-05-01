/*
 * File              : 1_1.cc
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2018.09.11
 * Last Modified Date: 2019.04.29
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */

/*  Find longest sequence of zeros in binary representation of an integer.
 *  https://app.codility.com/programmers/lessons/1-iterations/binary_gap/
 */

#include <iostream>

class Solution {
 public:
  int solution(int N) {
    int zeros = -1;
    int max_gap = 0;

    while (N > 0) {
      if (N & 0x1) {
        max_gap = std::max(max_gap, zeros);
        zeros = 0;
      } else if (zeros >= 0) {
        zeros++;
      }
      N = N >> 1;
    }

    return max_gap;
  }
};

int main(void) {
  Solution s;
  int longest = 0;

  longest = s.solution(1041);
  std::cout << longest << std::endl;

  longest = s.solution(32);
  std::cout << longest << std::endl;

  longest = s.solution(0);
  std::cout << longest << std::endl;

  longest = s.solution(1);
  std::cout << longest << std::endl;

  longest = s.solution(15);
  std::cout << longest << std::endl;

  return 0;
}
