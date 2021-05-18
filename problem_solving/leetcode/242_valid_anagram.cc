#include <iostream>

class Solution {
public:
  bool isAnagram(std::string s, std::string t) {
    if (s.length() != t.length()) return false;

    int freq[26] = {0};

    for (int i = 0; i < s.length(); i++) {
      freq[s[i] - 'a']++;
      freq[t[i] - 'a']--;
    }

    for (int c : freq) {
      if (c != 0) {
        return false;
      }
    }
    return true;
  }
};

int main(void) {
  Solution s;

  bool check;

  check = s.isAnagram("anagram", "anagram");
  std::cout << check << std::endl;

  check = s.isAnagram("anagram", "agramana");
  std::cout << check << std::endl;

  return 0;
}
