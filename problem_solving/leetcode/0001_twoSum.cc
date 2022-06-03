/**
 * File              : 1_two_sum.cc
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2021.05.23
 * Last Modified Date: 2021.05.23
 * Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>
 */
#include <iostream>
#include <unordered_map>
#include <vector>

class Solution {
public:
  std::vector<int> twoSum(std::vector<int>& nums, int target) {
    std::unordered_map<int, int> map_;
    std::vector<int> result(2);
    for (int i = 0; i < nums.size(); i++) {
      if (map_.find(target - nums[i]) != map_.end()) {
        result[0] = map_.at(target - nums[i]);
        result[1] = i;
        break;
      }
      map_.emplace(nums[i], i);
    }
    return result;
  }
};

int main(void) {
  Solution s;
  std::vector<int> nums{2, 7, 11, 15};

  std::vector<int> result = s.twoSum(nums, 9);
  for (auto it : result) {
    std::cout << it << ", ";
  }
  std::cout << std::endl;

  return 0;
}
