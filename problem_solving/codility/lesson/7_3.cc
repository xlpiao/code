/**
 * File              : 7_3.cc
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2019.05.06
 * Last Modified Date: 2019.05.08
 * Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>
 */

/* Description:
 * https://app.codility.com/programmers/lessons/7-stacks_and_queues/nesting/
 */

#include <iostream>
#include <vector>

class Solution {
 public:
  int solution(std::string &S) {
    int parentheses = 0;

    for (int i = 0; i < S.size(); i++) {
      if (S[i] == '(') {
        parentheses++;
      } else {
        parentheses--;
        if (parentheses < 0) {
          return 0;
        }
      }
    }
    return (parentheses == 0);
  }
};

int main(void) {
  Solution s;

  std::string a = "(()(())())";
  std::cout << s.solution(a) << std::endl;

  std::string b = "())";
  std::cout << s.solution(b) << std::endl;

  return 0;
}
