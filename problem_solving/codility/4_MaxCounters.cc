/**
 * File              : 4_3.cc
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2019.05.01
 * Last Modified Date: 2019.05.06
 * Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>
 */

/* calculate the values of counters after applying all alternating operations:
 * increase counter by 1; set value of all counters to current maximum.
 * https://app.codility.com/programmers/lessons/4-counting_elements/max_counters/
 */

#include <iostream>
#include <vector>

using namespace std;

class Solution {
 public:
  std::vector<int> solution(int N, std::vector<int> &A) {
#if 1
    int n = A.size();
    vector<int> counter(N);

    int maxCounter = 0;
    int currentMaxCount = 0;
    for (int i = 0; i < n; i++) {
      if (A[i] == N + 1) {
        maxCounter = currentMaxCount;
      } else {
        int idx = A[i] - 1;
        counter[idx] =
            (maxCounter > counter[idx]) ? maxCounter + 1 : counter[idx] + 1;
        currentMaxCount = max(currentMaxCount, counter[idx]);
      }
    }
    for (int i = 0; i < N; i++) {
      counter[i] = max(counter[i], maxCounter);
    }
    return counter;
#else  // time limit for large input
    int n = A.size();
    vector<int> counter(N);

    int maxCounter = 0;
    for (int i = 0; i < n; i++) {
      if (A[i] == N + 1) {
        counter.assign(N, maxCounter);
      } else {  // if(A[i] >= 1 && A[i] <= N) {
        counter[A[i] - 1]++;
        maxCounter = std::max(maxCounter, counter[A[i] - 1]);
      }
    }
    return counter;
#endif
  }
};

int main(void) {
  Solution s;

  std::vector<int> a = {3, 4, 4, 6, 1, 4, 4};
  std::vector<int> result = s.solution(5, a);

  for (int i = 0; i < result.size(); i++) {
    std::cout << "result: " << result[i] << std::endl;
  }

  return 0;
}
