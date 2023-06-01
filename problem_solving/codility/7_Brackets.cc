/**
 * File              : 7_1.cc
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2019.05.06
 * Last Modified Date: 2019.05.06
 * Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>
 */

/* Description:
 * https://app.codility.com/programmers/lessons/7-stacks_and_queues/brackets/ */

#include <iostream>
#include <stack>
#include <string>
#include <vector>

using namespace std;

class Solution {
 public:
  int solution(string &S) {
    if (S.empty()) return 1;
    int n = S.size();
    if (n % 2 != 0) return 0;

    std::stack<int> st;
    for (int i = 0; i < n; i++) {
      char c = S[i];
      if (c == '{' || c == '[' || c == '(') {
        st.push(c);
        continue;
      }

      char top = st.top();
      if (c == ')') {
        if (st.empty() || top != '(') {
          return false;
        }
        st.pop();
      } else if (c == ']') {
        if (st.empty() || top != '[') {
          return false;
        }
        st.pop();
      } else if (c == '}') {
        if (st.empty() || top != '{') {
          return false;
        }
        st.pop();
      }
    }
    return st.empty();
  }
};

int main(void) {
  Solution s;

  std::string a = "{[()()]}";
  std::cout << s.solution(a) << std::endl;

  std::string b = "([)()]";
  std::cout << s.solution(b) << std::endl;

  return 0;
}
