/**
 * File              : 7_4.cc
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2019.05.06
 * Last Modified Date: 2019.05.08
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */

/* Description:
 * https://app.codility.com/programmers/lessons/7-stacks_and_queues/nesting/
 */

#include <iostream>
#include <stack>
#include <vector>

class Solution {
 public:
  int solution(std::vector<int> &H) {
    std::stack<int> s;
    int blocks = 0;
    for (unsigned int i = 0; i < H.size(); i++) {
      while (!s.empty() && s.top() > H[i]) {
        s.pop();
        blocks++;
      }
      if (s.empty() || s.top() < H[i]) {
        s.push(H[i]);
      }
    }
    blocks += s.size();
    return blocks;
  }
};

int main(void) {
  Solution s;

  std::vector<int> h = {8, 8, 5, 7, 9, 8, 7, 4, 8};
  std::cout << s.solution(h) << std::endl;

  return 0;
}
