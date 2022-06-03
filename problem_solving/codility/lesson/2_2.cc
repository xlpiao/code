/**
 * File              : 2_2.cc
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2019.04.29
 * Last Modified Date: 2019.04.29
 * Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>
 */

/* Rotate an array to the right by a given number of steps.
 * https://app.codility.com/programmers/lessons/2-arrays/cyclic_rotation/  */

#include <iostream>
#include <vector>

class Solution {
 public:
  std::vector<int> solution(std::vector<int> &A, int K) {
    int N = A.size();
    if (N != 0) {
      for (unsigned int i = 0; i < K; i++) {
        int temp = A[N - 1];
        for (unsigned int j = N - 1; j > 0; j--) {
          A[j] = A[j - 1];
        }
        A[0] = temp;
      }
    }
    return A;
  }
};

int main(void) {
  Solution s;

  std::vector<int> A{3, 8, 9, 7, 6};
  std::vector<int> B{0, 0, 0};
  std::vector<int> C{1, 2, 3, 4};
  std::vector<int> D;

  std::vector<int> AA = s.solution(A, 3);
  std::vector<int> BB = s.solution(B, 1);
  std::vector<int> CC = s.solution(C, 4);
  std::vector<int> DD = s.solution(D, 3);

  for (auto elem : AA) {
    std::cout << elem << ",";
  }
  std::cout << "\n";

  for (auto elem : BB) {
    std::cout << elem << ",";
  }
  std::cout << "\n";

  for (auto elem : CC) {
    std::cout << elem << ",";
  }
  std::cout << "\n";

  for (auto elem : DD) {
    std::cout << elem << ",";
  }
  std::cout << "\n";

  return 0;
}
