#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  void dfs(vector<int>& candidates,
           int target,
           int sum,
           int start,
           vector<int>& subset,
           vector<vector<int>>& ans) {
    if (sum == target) {
      for (auto it : subset) {
        printf("%d, ", it);
      }
      printf("\n");

      ans.push_back(subset);
      return;
    }
    if (sum > target) {
      return;
    }

    for (int i = start; i < candidates.size(); i++) {
      if (i > start && candidates[i] == candidates[i - 1]) {
        continue;
      }
      subset.push_back(candidates[i]);
      dfs(candidates, target, sum + candidates[i], i + 1, subset, ans);
      subset.pop_back();
    }
  }

  vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
    sort(candidates.begin(), candidates.end());

    vector<int> subset;
    vector<vector<int>> ans;

    dfs(candidates, target, 0, 0, subset, ans);

    return ans;
  }
};

int main(void) {
  Solution s;
  vector<int> nums = {2, 5, 2, 1, 2};

  auto ans = s.combinationSum2(nums, 5);

  return 0;
}
