#include <iostream>
#include <vector>

class Solution {
public:
  int removeDuplicates(std::vector<int>& nums) {
    if (nums.size() == 0) {
      return 0;
    }
    for (int i = 0; i < nums.size() - 1; i++) {
      if (nums[i] == nums[i + 1]) {
        nums.erase(nums.begin() + i + 1);
        i--;
      }
    }
    return nums.size();
  }
};

int main(void) {
  Solution s;

  std::vector<int> nums{0, 0, 1, 1, 1, 2, 2, 3, 3, 4};
  int ret = s.removeDuplicates(nums);
  std::cout << ret << std::endl;
}
