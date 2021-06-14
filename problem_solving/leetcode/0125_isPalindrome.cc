#include <iostream>
#include <string>

using namespace std;

class Solution {
public:
  bool isPalindrome(string s) {
    int left = 0;
    int right = s.size() - 1;
    while (left <= right) {
      if (!isalnum(s[left])) {
        left++;
        continue;
      }
      if (!isalnum(s[right])) {
        right--;
        continue;
      }
      if (tolower(s[left]) != tolower(s[right])) {
        return false;
      }
      left++;
      right--;
    }
    return true;
  }
};

int main(void) {
  Solution s;
  cout << s.isPalindrome("A man, a plan, a canal: Panama") << endl;
  cout << s.isPalindrome("race a car") << endl;
}
