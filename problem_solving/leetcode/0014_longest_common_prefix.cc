#include <iostream>
#include <string>
#include <vector>

class Solution {
public:
  std::string longestCommonPrefix(std::vector<std::string>& strs) {
    if (strs.size() == 0) return "";

    for (int i = 0; i < strs[0].size(); i++) {
      for (int j = 1; j < strs.size(); j++) {
        if (strs[j][i] != strs[0][i] || i >= strs[j].size()) {
          return strs[0].substr(0, i);
        }
      }
    }

    return strs[0];
  }
};

int main(void) {
  Solution s;

  std::vector<std::string> strs{"flower", "flow", "flight"};

  std::string ret = s.longestCommonPrefix(strs);

  std::cout << ret << std::endl;
}
