class Solution {
public:
  int strStr(string haystack, string needle) {
    int m = haystack.size();
    int n = needle.size();

    if (needle.empty()) {
      return 0;
    }

    int i, j;
    for (i = 0; i < m; i++) {
      if (haystack.size() - i < n) return -1;

      for (j = 0; j < n; j++) {
        if (haystack[j + i] != needle[j]) {
          break;
        }
      }
      if (j == n) {
        return i;
      }
    }

    return -1;
  }
};
