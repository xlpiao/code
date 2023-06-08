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

#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>
using namespace std;

int solution(vector<int> &A) {
  int N = A.size();
  vector<long> dp(N, numeric_limits<long>::min());
  dp[0] = A[0];
  for (int i = 1; i < N; ++i) {
    int die = 1;
    while (die <= 6 && i - die >= 0) {
      dp[i] = std::max(dp[i], A[i] + dp[i - die]);
      ++die;
    }
  }

  // for (auto it : dp) {
  // cout << it << ", ";
  // }
  return dp[N - 1];
}

int main(void) {
  std::vector<int> A = {1, -2, 0, 9, -1, -2};
  auto ans = solution(A);

  std::cout << ans << std::endl;

  return 0;
}
