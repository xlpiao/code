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
      if (start != i && nums[start] == nums[i]) continue;
      swap(nums[start], nums[i]);
      dfs(nums, start + 1, ans);
      // swap(nums[start], nums[i]);
    }
  }

  vector<vector<int>> permuteUnique(vector<int>& nums) {
    vector<vector<int>> ans;

    sort(nums.begin(), nums.end());
    dfs(nums, 0, ans);

    return ans;
  }
};

int main(void) {
  Solution s;
  vector<int> nums = {1, 1, 2};

  s.permuteUnique(nums);

  return 0;
}
