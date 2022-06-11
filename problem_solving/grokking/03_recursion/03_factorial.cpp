#include <iostream>

using std::cout;
using std::endl;

int fact(const int& x) {
  if (x == 1) return 1;
  return fact(x - 1) * x;
}

int main(void) {
  cout << fact(5) << endl;
  return 0;
}
