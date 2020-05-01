/**
 * File              : t03_combination.c
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2018.08.13
 * Last Modified Date: 2018.08.23
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */

#include <stdio.h>

int solution(int N) {
  int digit_count[10] = {0};
  int num_digits = 0;
  int num_zeros = 0;
  int num_similars = 1;

  for (int i = N; i > 0; i /= 10) {
    int digit = i % 10;
    digit_count[digit]++;
    num_digits++;
  }
  num_zeros = digit_count[0];

  for (int i = 0; i < 10; i++) {
    printf("Number of '%d':\t%d\n", i, digit_count[i]);
  }
  printf("num_zeros    :\t%d\n", num_zeros);
  printf("num_digits   :\t%d\n", num_digits);

  num_similars = num_digits - num_zeros;
  printf("num_similars :\t%d\n", num_similars);
  for (int i = num_digits - 1; i > 1; i--) {
    num_similars *= i;
    printf("%d-th num_similars :\t%d\n", i, num_similars);
  }

  for (int i = 0; i < 10; i++) {
    for (int j = digit_count[i]; j > 1; j--) {
      num_similars /= j;
      printf("%d-th num_similars :\t%d\n", j, num_similars);
    }
  }

  return (num_similars > 0) ? num_similars : 1;
}

int main(void) {
  int A = 1213;
  printf("The number of arrangement of %d: %d\n", A, solution(A));  // 12

  int B = 123;
  printf("The number of arrangement of %d: %d\n", B, solution(B));  // 6

  int C = 100;
  printf("The number of arrangement of %d: %d\n", C, solution(C));  // 1

  int D = 0;
  printf("The number of arrangement of %d: %d\n", D, solution(D));  // 1

  int E = 1230;
  printf("The number of arrangement of %d: %d\n", E, solution(E));  // 18

  int F = 1113;
  printf("The number of arrangement of %d: %d\n", F, solution(F));  // 4

  int G = 11223;
  printf("The number of arrangement of %d: %d\n", G, solution(G));  // 30

  int H = 11220;
  printf("The number of arrangement of %d: %d\n", H, solution(H));  // 24

  return 0;
}
