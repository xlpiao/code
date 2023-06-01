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
    int N = A.size();
    if (N <= 1) return N;

    int alive = N;
    std::stack<int> st;

    for (int i = 0; i < N; i++) {
      if (B[i] == 1) {
        st.push(i);
      } else {
        while (!st.empty() && A[st.top()] < A[i]) {
          st.pop();
          --alive;
        }
        if (!st.empty() && A[st.top()] > A[i]) {
          --alive;
        }
      }
    }
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
