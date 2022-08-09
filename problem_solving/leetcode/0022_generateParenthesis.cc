/**
 * File              : 0022_generateParenthesis.cc
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2022.08.09
 * Last Modified Date: 2022.08.09
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

class Solution {
public:
  std::vector<std::string> generateParenthesis(int n) {
    std::vector<std::string> ans;
    generate(ans, "", n, n);
    return ans;
  }

  void generate(std::vector<std::string> &ans,
                std::string s,
                int left,
                int right) {
    if (left == 0 && right == 0) {
      ans.push_back(s);
      return;
    }

    if (left > 0) {
      generate(ans, s + '(', left - 1, right);
    }

    if (right > 0 && left < right) {
      generate(ans, s + ')', left, right - 1);
    }
  }
};

int main(void) {
  Solution s;
  auto result = s.generateParenthesis(3);
  for (auto it : result) {
    std::cout << it << std::endl;
  }

  return 0;
}
