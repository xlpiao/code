/**
 * File              : 7_2.cc
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2019.05.06
 * Last Modified Date: 2019.05.07
 * Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>
 */

/* Description:
 * https://app.codility.com/programmers/lessons/7-stacks_and_queues/fish/
 */

#include <iostream>
#include <stack>
#include <vector>

class Solution {
 public:
  int solution(std::vector<int> &A, std::vector<int> &B) {
    std::stack<int> s;
    int N = A.size();
    int alive = 0;
    for (int i = 0; i < N; i++) {
      if (B[i] == 1) {
        s.push(A[i]);
      } else {
        while (!s.empty() && (s.top() < A[i])) {
          s.pop();
        }
        if (s.empty()) {
          alive++;
        }
      }
    }
    alive += s.size();
    return alive;
  }
};

int main(void) {
  Solution s;

  std::vector<int> a = {4, 3, 2, 1, 5};
  std::vector<int> b = {0, 1, 0, 0, 0};
  std::cout << s.solution(a, b) << std::endl;

  return 0;
}
