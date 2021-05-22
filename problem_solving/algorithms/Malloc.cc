#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
  char *str;

  /* malloc() allocate the memory for 5 integers */
  /* containing garbage values */
  str = (char *)malloc(15 * sizeof(char));  // 5*4bytes = 5 bytes

  /* Deallocates memory previously allocated by malloc() function */
  free(str);

  /* calloc() allocate the memory for 5 integers and */
  /* set 0 to all of them */
  str = (char *)calloc(15, sizeof(char));

  strcpy(str, "tutorialspoint");
  printf("String = %s\n", str);

  /* Reallocating memory */
  /*
   * if str is null, malloc
   * else, resize allocation.
   */
  str = (char *)realloc(str, 30);
  strcat(str, ".com");
  printf("String = %s\n", str);

  free(str);

  return 0;
}
