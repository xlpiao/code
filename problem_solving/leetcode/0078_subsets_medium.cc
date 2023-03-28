#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  void backtrack(int start,
                 int n,
                 vector<int>& nums,
                 vector<int>& subset,
                 vector<vector<int>>& ans) {
    ans.push_back(subset);

    for (int i = start; i < n; i++) {
      subset.push_back(nums[i]);
      backtrack(i + 1, n, nums, subset, ans);
      subset.pop_back();
    }
  }

  vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> ans;
    vector<int> subset;
    backtrack(0, nums.size(), nums, subset, ans);

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
