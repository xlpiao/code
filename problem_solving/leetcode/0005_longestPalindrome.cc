#include <iostream>

class Solution {
public:
  std::string longestPalindrome(std::string s) {
    int start = 0;
    int maxlen = 1;

    for (int i = 0; i < s.size(); i++) {
      // Case #1: *cdabba*
      // Case #1: *012345*
      // 0,1
      // 1,2/0,3
      // 2,3/1,4/0,5
      int left1 = i;
      int right1 = i + 1;
      int len1 = 0;
      while (left1 >= 0 && right1 < s.size() && s[left1] == s[right1]) {
        len1 += 2;
        left1--;
        right1++;
      }
      if (maxlen < len1) {
        left1++;
        right1--;
        start = left1;
        maxlen = len1;
      }

      // Case #2: *cdaba*
      // Case #2: *01234*
      // 0
      // 1/0,2
      // 2/1,3/0,4
      int middle = i + 1;
      int left2 = middle - 1;
      int right2 = middle + 1;
      int len2 = 1;
      while (left2 >= 0 && right2 < s.size() && s[left2] == s[right2]) {
        len2 += 2;
        left2--;
        right2++;
      }
      if (maxlen < len2) {
        left2++;
        right2--;
        start = left2;
        maxlen = len2;
      }
    }
    return s.substr(start, maxlen);
  }

private:
  bool isPalindrom(std::string s) {
    for (int i = 0; i < s.size() / 2; i++) {
      if (s[i] != s[s.size() - i - 1]) {
        return false;
      }
    }
    return true;
  }
};

int main(void) {
  Solution s;
  std::string ret;

  ret = s.longestPalindrome("cdabba");
  std::cout << "cdabba: " << ret << std::endl;

  ret = s.longestPalindrome("aba");
  std::cout << "aba: " << ret << std::endl;

  ret = s.longestPalindrome("cbb");
  std::cout << "cbb: " << ret << std::endl;

  ret = s.longestPalindrome("ccc");
  std::cout << "ccc: " << ret << std::endl;
}
