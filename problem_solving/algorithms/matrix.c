#include <stdio.h>
#include <stdlib.h>  // rand(), srand()
#include <time.h>    // time()

#define M 4
#define N 3

#define MIN_NUM 5
#define MAX_NUM 10

void genMat(int **mat, int row, int col) {
  /* Generate random integer between MIN_NUM and MAX_NUM */
  srand(time(NULL));
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      /* NOTE: random int between 0 and 19 */
      // int r = rand() % 20;
      mat[i][j] = rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM;
      // *(*(mat + i) + j) = rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM;
    }
  }
}

void printMat(int **mat, int row, int col) {
  printf("mat[%d][%d] = {\n", row, col);
  for (int i = 0; i < row; i++) {
    printf("{");
    for (int j = 0; j < col; j++) {
      if (j < col - 1)
        // printf("%2d, ", *(*(mat + i) + j));
        printf("%2d, ", mat[i][j]);
      else
        // printf("%2d", *(*(mat + i) + j));
        printf("%2d", mat[i][j]);
    }
    if (i < row - 1)
      printf("},\n");
    else
      printf("} ");
  }
  printf("}\n\n");
}

int main(void) {
  int **mat1 = (int **)malloc(M * sizeof(int *));
  for (int i = 0; i < M; i++) mat1[i] = (int *)malloc(N * sizeof(int));

  int **mat2 = (int **)malloc(M * sizeof(int *));
  for (int j = 0; j < N; j++) mat2[j] = (int *)malloc(M * sizeof(int));

  /* Generate 2D matrix */
  genMat(mat1, M, N);
  genMat(mat2, N, M);

  /* Print the generated random number */
  printMat(mat1, M, N);
  printMat(mat2, N, M);

  for (int i = 0; i < M; i++) free(mat1[i]);
  free(mat1);

  for (int j = 0; j < N; j++) free(mat2[j]);
  free(mat2);

  return 0;
}
