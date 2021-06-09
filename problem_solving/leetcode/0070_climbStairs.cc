#include <iostream>

using namespace std;

class Solution {
public:
  int climbStairs(int n) {
    int dp[n + 1];
    dp[0] = 1;
    dp[1] = 1;

    for (int i = 2; i <= n; i++) {
      dp[i] = dp[i - 1] + dp[i - 2];
    }

    return dp[n];
  }
};

int main(void) {
  Solution s;
  int n;

  n = 6;
  cout << n << ": " << s.climbStairs(n) << " ways" << endl;

  n = 45;
  cout << n << ": " << s.climbStairs(n) << " ways" << endl;

  return 0;
}
