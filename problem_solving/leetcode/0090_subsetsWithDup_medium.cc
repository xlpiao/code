#include <iostream>
#include <vector>

using namespace std;

class Solution {
 public:
  void backtrack(vector<int>& nums, int start, int end, vector<int>& subset,
                 vector<vector<int>>& ans) {
    ans.push_back(subset);

    for (int i = start; i < end; i++) {
      if (i > start && nums[i] == nums[i - 1]) {
        continue;
      }
      subset.push_back(nums[i]);
      backtrack(nums, i + 1, end, subset, ans);
      subset.pop_back();
    }
  }

  vector<vector<int>> subsetsWithDup(vector<int>& nums) {
    sort(nums.begin(), nums.end());

    vector<vector<int>> ans;
    vector<int> subset;
    backtrack(nums, 0, nums.size(), subset, ans);
    // sort(ans.begin(), ans.end());
    // ans.erase(unique(ans.begin(), ans.end()), ans.end());

    return ans;
  }
};

int main(void) {
  Solution s;
  vector<int> nums = {3, 5, 5};

  cout << "input: ";
  for (auto it : nums) {
    cout << it << ", ";
  }
  cout << endl;

  auto ans = s.subsetsWithDup(nums);

  cout << "output: ";
  for (auto it : ans) {
    for (auto elem : it) {
      cout << elem << ", ";
    }
    cout << endl;
  }

  return 0;
}
