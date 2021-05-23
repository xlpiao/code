/**
 * File              : 7_reverse_integer.cc
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2021.05.23
 * Last Modified Date: 2021.05.23
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */

#include <iostream>

class Solution {
public:
  int reverse(int x) {
    int rev = 0;
    while (x != 0) {
      rev = rev * 10 + x % 10;
      x = x / 10;
    }
    return rev;
  }
};

int main(void) {
  Solution s;
  int result = 0;

  result = s.reverse(123);
  std::cout << "Reverse 123: " << result << std::endl;

  result = s.reverse(1100);
  std::cout << "Reverse 1100: " << result << std::endl;

  return 0;
}
