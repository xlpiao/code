#include <iostream>
#include <vector>

using namespace std;

class Solution {
 public:
  void backtrack(vector<int>& nums, int start, int end, vector<int>& subset,
                 vector<vector<int>>& ans) {
    ans.push_back(subset);

    for (int i = start; i < end; i++) {
      subset.push_back(nums[i]);
      backtrack(nums, i + 1, end, subset, ans);
      subset.pop_back();
    }
  }

  vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> ans;
    vector<int> subset;
    backtrack(nums, 0, nums.size(), subset, ans);

    return ans;
  }
};

int main(void) {
  Solution s;
  vector<int> nums = {3, 4, 5};

  cout << "input: ";
  for (auto it : nums) {
    cout << it << ", ";
  }
  cout << endl;

  auto ans = s.subsets(nums);

  cout << "output: ";
  for (auto it : ans) {
    for (auto elem : it) {
      cout << elem << ", ";
    }
    cout << endl;
  }

  return 0;
}
