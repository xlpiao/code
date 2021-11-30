#include <iostream>

class Solution {
public:
  int fib(int n) {
    int size = n + 2;
    int f[size];

    f[0] = 0;
    f[1] = 1;

    for (int i = 2; i <= n; i++) {
      f[i] = f[i - 1] + f[i - 2];
    }

    return f[n];
  }
};

int main(void) {
  std::cout << fib(10) << std::endl;
  return 0;
}
