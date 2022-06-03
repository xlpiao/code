/**
 * File              : t01_die.c
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2018.08.13
 * Last Modified Date: 2018.08.26
 * Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>
 */

#include <stdio.h>

int solution(int A[], int N) {
  int min_num_moves = N * 2;

  for (int i = 0; i < N; i++) {
    int target_pip = A[i];
    int num_moves = 0;
    for (int j = 0; j < N; j++) {
      if (i != j) {
        if (A[j] != target_pip) {
          num_moves++;
        }
        if (A[j] + target_pip == 7) {
          num_moves++;
        }
      }
    }

    if (min_num_moves > num_moves) {
      min_num_moves = num_moves;
    }
  }

  return min_num_moves;
}

int main(void) {
  int A[] = {1, 2, 3};
  printf("%d\n", solution(A, 3));

  int B[] = {1, 1, 6};
  printf("%d\n", solution(B, 3));

  int C[] = {1, 6, 2, 3};
  printf("%d\n", solution(C, 4));

  return 0;
}
