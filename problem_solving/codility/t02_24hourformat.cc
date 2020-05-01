/**
 * File              : t02_24hourformat.c
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2018.08.13
 * Last Modified Date: 2018.08.23
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* solution(int A, int B, int C, int D, int E, int F) {
  // write your code in C99 (gcc 6.2.0)
  int N = 6;
  int input[] = {A, B, C, D, E, F};
  char* hhmmss;
  char* notpossible = "NOT POSSIBLE";

  hhmmss = (char*)calloc(16, sizeof(char));

  // sorting
  int temp;
  for (int i = 0; i < N - 1; i++) {
    for (int j = i + 1; j < N; j++) {
      if (input[i] > input[j]) {
        temp = input[i];
        input[i] = input[j];
        input[j] = temp;
      }
    }
  }

  // 24-hour format
  if (input[4] < 6) {
    if (10 * input[0] + input[1] < 24) {
      hhmmss[0] = input[0] + '0';
      hhmmss[1] = input[1] + '0';
      hhmmss[2] = ':';
      hhmmss[3] = input[2] + '0';
      hhmmss[4] = input[3] + '0';
      hhmmss[5] = ':';
      hhmmss[6] = input[4] + '0';
      hhmmss[7] = input[5] + '0';
      hhmmss[8] = '\0';
      return hhmmss;
    }
  } else if (input[3] < 6) {
    if (10 * input[0] + input[1] < 24) {
      hhmmss[0] = input[0] + '0';
      hhmmss[1] = input[1] + '0';
      hhmmss[2] = ':';
      hhmmss[3] = input[2] + '0';
      hhmmss[4] = input[4] + '0';
      hhmmss[5] = ':';
      hhmmss[6] = input[3] + '0';
      hhmmss[7] = input[5] + '0';
      hhmmss[8] = '\0';
      return hhmmss;
    }
  } else if (input[2] < 6) {
    if (10 * input[0] + input[3] < 24) {
      hhmmss[0] = input[0] + '0';
      hhmmss[1] = input[3] + '0';
      hhmmss[2] = ':';
      hhmmss[3] = input[1] + '0';
      hhmmss[4] = input[4] + '0';
      hhmmss[5] = ':';
      hhmmss[6] = input[2] + '0';
      hhmmss[7] = input[5] + '0';
      hhmmss[8] = '\0';
      return hhmmss;
    }
  }

  strcpy(hhmmss, notpossible);
  return hhmmss;
}

int main(void) {
  char* b = solution(2, 4, 5, 9, 5, 9);
  for (int i = 0; i < 12; i++) {
    printf("%c", b[i]);
  }
  printf("\n");

  char* c = solution(9, 7, 0, 0, 0, 8);
  for (int i = 0; i < 12; i++) {
    printf("%c", c[i]);
  }
  printf("\n");

  char* d = solution(1, 8, 3, 2, 6, 4);
  for (int i = 0; i < 12; i++) {
    printf("%c", d[i]);
  }
  printf("\n");

  return 0;
}
