/**
 * File              : 7_1.cc
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2019.05.06
 * Last Modified Date: 2019.05.06
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */

/* Description:
 * https://app.codility.com/programmers/lessons/7-stacks_and_queues/brackets/ */

#include <iostream>
#include <stack>
#include <vector>

class Solution {
 public:
  int solution(std::string &S) {
    std::stack<int> s;
    for (int i = 0; i < S.size(); i++) {
      char c = S[i];
      char t;
      if (c == '{' || c == '[' || c == '(') {
        s.push(c);
        continue;
      }

      if (s.empty()) return false;

      if (c == ')') {
        t = s.top();
        s.pop();
        if (t == '{' || t == '[') {
          return false;
        }
      } else if (c == ']') {
        t = s.top();
        s.pop();
        if (t == '{' || t == '(') {
          return false;
        }
      } else if (c == '}') {
        t = s.top();
        s.pop();
        if (t == '[' || t == '(') {
          return false;
        }
      }
    }
    return (s.empty());
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
