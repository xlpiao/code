/**
 * File              : 3_1.cc
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2019.04.29
 * Last Modified Date: 2019.04.29
 * Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>
 */

/* Count minimal number of jumps from position X to Y.
 * https://app.codility.com/programmers/lessons/3-time_complexity/frog_jmp/
 */

#include <iostream>
using namespace std;

class Solution {
 public:
  int solution(int X, int Y, int D) {
    int remainder = (Y - X) % D;
    int quotient = (Y - X) / D;
    if (remainder == 0) {
      return quotient;
    } else {
      return quotient + 1;
    }
  }
};

int main(void) {
  Solution s;

  cout << s.solution(10, 85, 30) << endl;

  return 0;
}
