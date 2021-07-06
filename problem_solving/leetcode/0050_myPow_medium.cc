class Solution {
public:
  double myPow(double x, int n) {
    if (n == 0) return 1;
    if (n == 1 || x == 1) return x;

    long pow = n;
    if (pow < 0) {
      pow = (-1) * pow;
    }

    double result = 1.0;
    while (pow) {
      if (pow % 2 == 0) {
        x = x * x;
        pow = pow / 2;
      } else {
        result = result * x;
        pow--;
      }
    }

    if (n < 0) result = (double)(1.0) / result;

    return result;
  }
};
