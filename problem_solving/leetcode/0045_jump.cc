#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  int jump(vector<int>& nums) {
    int len = nums.size();

    vector<int> dp(len, 10001);

    for (int i = 0; i < len; i++) {
      for (int j = i; j <= i + nums[i] && j < len; j++) {
        dp[j] = min(i, dp[j]);
      }
    }

    int jump = 0;
    int idx = len - 1;
    while (idx > 0) {
      idx = dp[idx];
      jump++;
    }

    return jump;
  }
};

#if 0
class Solution {
public:
  int jump(vector<int>& nums) {
    if (nums.size() == 0) {
      return 0;
    }

    int jump = 0;
    int next = 0;
    int curr = 0;
    for (int i = 0; i < nums.size(); i++) {
      if (curr < i) {
        jump++;
        curr = next;
      }

      next = max(next, i + nums[i]);
    }
    return jump;
  }
};
#endif

int main(void) {
  Solution s;

  vector<int> nums{2, 3, 0, 1, 4};
  cout << s.jump(nums) << endl;

  return 0;
}
