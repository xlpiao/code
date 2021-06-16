#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  int singleNumber(vector<int>& nums) {
    int ret = nums[0];
    for (int i = 1; i < nums.size(); i++) {
      ret ^= nums[i];
    }
    return ret;
  }
};

int main(void) {
  Solution s;

  vector<int> nums{4, 2, 1, 2, 1};
  cout << s.singleNumber(nums) << endl;

  return 0;
}
