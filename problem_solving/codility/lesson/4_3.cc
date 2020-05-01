/**
 * File              : 4_3.cc
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2019.05.01
 * Last Modified Date: 2019.05.06
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */

/* calculate the values of counters after applying all alternating operations:
 * increase counter by 1; set value of all counters to current maximum.
 * https://app.codility.com/programmers/lessons/4-counting_elements/max_counters/
 */

#include <iostream>
#include <vector>

class Solution {
 public:
  std::vector<int> solution(int N, std::vector<int> &A) {
    int max_count = 0;
    std::vector<int> bitmap(N);
    // std::vector<int> bitmap(N, 0); // same

    for (unsigned int i = 0; i < A.size(); i++) {
      if (A[i] == N + 1) {
        bitmap.assign(N, max_count);
        // std::fill(bitmap.begin(), bitmap.end(), max_count);
      } else {
        bitmap[A[i] - 1]++;
        max_count = std::max(max_count, bitmap[A[i] - 1]);
      }
    }
    return bitmap;
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
