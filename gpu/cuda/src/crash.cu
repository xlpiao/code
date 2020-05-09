#include <stdio.h>
#include <stdlib.h>

__global__ void foo(int *ptr) { *ptr = 7; }

int main(void) {
  foo<<<1, 1>>>(0);
  return 0;
}
