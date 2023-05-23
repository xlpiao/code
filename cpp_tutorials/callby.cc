#include <iostream>

void swapByValue(int x, int y) {
  int temp = x;
  x = y;
  y = temp;
}

void swapByReference(int& x, int& y) {
  int temp = x;
  x = y;
  y = temp;
}

void swapByAddress(int* x, int* y)  // Call by pointer or Call by address
{
  int temp = *x;
  *x = *y;
  *y = temp;
}

int main(void) {
  int x1 = 1, y1 = 2;
  swapByValue(x1, y1);
  std::cout << "x1 = " << x1 << ", y1 = " << y1 << std::endl;

  int x2 = 1, y2 = 2;
  swapByReference(x2, y2);
  std::cout << "x2 = " << x2 << ", y2 = " << y2 << std::endl;

  int x3 = 1, y3 = 2;
  swapByAddress(&x3, &y3);
  std::cout << "x3 = " << x3 << ", y3 = " << y3 << std::endl;

  return 0;
}
