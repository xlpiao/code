#include <iostream>

class Solution {
public:
  std::string longestPalindrome(std::string s) {
    int maxlen = 0;
    int start = 0;

    for (int i = 0; i < s.size(); i++) {
      for (int j = i; j < s.size(); j++) {
        int left = i;
        int right = j - i + 1;
        // std::cout << s.substr(left, right) << std::endl;
        if (isPalindrom(s.substr(left, right))) {
          int len = s.substr(left, right).size();
          if (maxlen < len) {
            maxlen = len;
            start = left;
          }
        }
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

  std::string ret1 = s.longestPalindrome("babad");
  std::cout << ret1 << std::endl;

  std::string ret2 = s.longestPalindrome("aba");
  std::cout << ret2 << std::endl;
}
