/**
 * File              : scanf.c
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2018.08.11
 * Last Modified Date: 2018.08.11
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */
#include <stdio.h>

#define N 6
char map[N][N];

int main (void) {
	int nRows, nCols;
    // int nWolves = 0;
    // int nGoats = 0;

    /* input */
    scanf("%d %d", &nRows, &nCols);
    for (int i = 0; i < nRows; i++) {
        scanf("%s", map[i]);

        // for (int j = 0; j < nCols; j++) {
            // printf("%c", map[i][j]);
        // }
        // printf("\n");

        printf("%s\n", map[i]);
    }


    return 0;
}
