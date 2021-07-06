class Solution {
public:
  int mySqrt(int x) {
    if (x == 0 || x == 1) {
      return x;
    }

    int left = 1, right = x;
    long long middle;
    while (left < right) {
      middle = left + (right - left) / 2;
      if (middle * middle == x) {
        return middle;
      } else if (middle * middle < x) {
        left = middle + 1;
      } else {
        right = middle;
      }
    }
    return left - 1;
  }
};
