#include <stdio.h>

#define N 251
char map[N][N];

int main (void) {
    int nWolves = 0;
    int nGoats = 0;
	int nRows, nCols;

    /* input */
    scanf("%d %d", &nRows, &nCols);
    for (int i = 0; i < nRows; i++) {
        scanf("%s", map[i]);

        for (int j = 0; j < nCols; j++) {
            printf("%c", map[i][j]);
        }
        printf("\n");

        // printf("%s\n", map[i]);
    }

    return 0;
}
