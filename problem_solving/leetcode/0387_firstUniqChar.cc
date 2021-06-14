#include <iostream>
#include <string>

using namespace std;

class Solution {
public:
  int firstUniqChar(string s) {
    int alphabet[26] = {0};

    for (int i = 0; i < s.size(); i++) {
      alphabet[s[i] - 'a']++;
    }
    for (int i = 0; i < s.size(); i++) {
      if (alphabet[s[i] - 'a'] == 1) {
        return i;
      }
    }
    return -1;
  }
};

int main(void) {
  Solution s;
  cout << s.firstUniqChar("leetcode") << endl;
  cout << s.firstUniqChar("loveleetcode") << endl;
}
