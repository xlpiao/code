#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
  void dfs(vector<int> nums, int start, vector<vector<int>>& ans) {
    if (start == nums.size() - 1) {
      for (auto it : nums) cout << it << ",";
      cout << endl;
      ans.push_back(nums);
      return;
    }

    for (int i = start; i < nums.size(); i++) {
      swap(nums[start], nums[i]);
      dfs(nums, start + 1, ans);
      // swap(nums[start], nums[i]);
    }
  }

  vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> ans;

    dfs(nums, 0, ans);

    return ans;
  }
};

int main(void) {
  Solution s;
  vector<int> nums = {1, 1, 2};

  s.permute(nums);

  return 0;
}
