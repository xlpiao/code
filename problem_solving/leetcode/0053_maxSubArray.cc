#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  int maxSubArray(vector<int>& nums) {
    int sum = 0;
    int maxsum = 0;
    for (int i = 0; i < nums.size(); i++) {
      if (sum < 0) {
        sum = 0;
      }
      sum += nums[i];
      if (maxsum < sum) {
        maxsum = sum;
      }
    }
    return maxsum;
  }
};

int main(void) {
  Solution s;

  vector<int> nums{-2, 1, -3, 4, -1, 2, 1, -5, 4};
  cout << s.maxSubArray(nums) << endl;

  return 0;
}
