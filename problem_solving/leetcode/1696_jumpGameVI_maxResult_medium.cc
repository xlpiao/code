#include <deque>
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  int maxResult(vector<int>& nums, int k) {
    int n = nums.size();

    vector<int> dp{nums};
    deque<int> q{0};

    for (int i = 1; i < n; i++) {
      dp[i] += dp[q.front()];

      while (!q.empty() && dp[i] > dp[q.back()]) {
        q.pop_back();
      }

      q.push_back(i);

      if (q.front() == i - k) {
        q.pop_front();
      }
    }

    return dp.back();
  }
};

int main(void) {
  Solution s;

  int result = 0;

  vector<int> nums1{10, -5, -2, 4, 0, 3};
  result = s.maxResult(nums1, 3);
  cout << "result: " << result << endl;

  vector<int> nums2{1, -5, -20, 4, -1, 3, -6, -3};
  result = s.maxResult(nums2, 2);
  cout << "result: " << result << endl;

  vector<int> nums3{100, -1, -100, -1, 100};
  result = s.maxResult(nums3, 2);
  cout << "result: " << result << endl;

  vector<int> nums4{-5582, -5317, 6711, -639, 1001, 1845, 1728, -4575};
  result = s.maxResult(nums4, 3);
  cout << "result: " << result << endl;

  return 0;
}
