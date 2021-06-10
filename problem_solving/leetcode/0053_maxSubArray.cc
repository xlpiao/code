#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  int maxSubArray(vector<int>& nums) {
    int maxsum = nums[0];
    int sum = 0;
    for (int i = 0; i < nums.size(); i++) {
      sum += nums[i];
      maxsum = max(sum, maxsum);
      sum = max(sum, 0);
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
