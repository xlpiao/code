class Solution {
public:
  int trailingZeroes(int n) {
    int count = 0;

    for (n = n / 5; n > 0; n /= 5) {
      count += n;
    }

    return count;
  }
};
