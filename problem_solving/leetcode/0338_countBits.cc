#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  vector<int> countBits(int n) {
    vector<int> dp(n + 1);
    int offset = 1;
    dp[0] = 0;

    for (int i = 1; i < n + 1; i++) {
      if (offset * 2 == i) {
        offset = i;
      }
      dp[i] = 1 + dp[i - offset];
    }
    return dp;
  }
};

int main(void) {
  Solution s;

  auto output = s.countBits(16);
  for (auto it : output) {
    cout << it << ", ";
  }
  cout << endl;
}
