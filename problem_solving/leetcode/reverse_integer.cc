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
