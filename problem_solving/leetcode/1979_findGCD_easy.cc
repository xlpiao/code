#include <iostream>
#include <vector>

class Solution {
public:
  // let c = a % b, then gcd(a, b) = gcd(a, c) = gcd(b, c)
  int gcd(int a, int b) {
    std::cout << a << "," << b << std::endl;
    if (b == 0) return a;
    return gcd(b, a % b);
  }
  int findGCD(std::vector<int>& nums) {
    int minNum = INT_MAX, maxNum = INT_MIN;
    for (int i = 0; i < nums.size(); i++) {
      minNum = std::min(minNum, nums[i]);
      maxNum = std::max(maxNum, nums[i]);
    }
    return gcd(maxNum, minNum);
  }
};

int main(void) {
  Solution s;

  std::vector<int> num{10, 4};

  std::cout << s.findGCD(num) << std::endl;

  return 0;
}
