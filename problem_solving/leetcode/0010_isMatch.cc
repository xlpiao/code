#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
  bool isMatch(string s, string p) {
    int sLen = s.size();
    int pLen = p.size();
    vector<vector<bool>> dp(pLen + 1, vector<bool>(sLen + 1, false));
    dp[0][0] = true;

    for (int i = 1; i < pLen + 1; i++) {
      if (p[i - 1] == '*') {
        dp[i][0] = dp[i - 2][0];
      }
      for (int j = 1; j < sLen + 1; j++) {
        if (p[i - 1] == '.' || p[i - 1] == s[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1];
        } else if (p[i - 1] == '*') {
          if (p[i - 2] == '.' || p[i - 2] == s[j - 1]) {
            dp[i][j] = dp[i][j - 1];
          }
          dp[i][j] = dp[i][j] || dp[i - 2][j];
        }
      }
    }
    return dp[pLen][sLen];
  }
};

int main(void) {
  Solution s;

  string input = "aebbbbc";
  // string input = "aec";
  string pattern = "aeb*c";

  bool ret = s.isMatch(input, pattern);
  cout << ret << endl;

  return 0;
}
