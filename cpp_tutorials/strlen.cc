#include <cstdio>
#include <cstdlib>

int strlen(char* str) {
  int count = 0;
  while (str[count] != '\0') {
    count++;
  }
  return count;
}

int main(void) {
  char* str;
  str = (char*)malloc(sizeof(char) * 6);
  for (int i = 0; i < 5; i++) {
    str[i] = i + 'a';
    printf("%c", str[i]);
  }
  str[5] = '\0';
  printf("\n");

  int len = strlen(str);
  printf("len: %d\n", len);
}
