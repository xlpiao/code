class Solution {
public:
  int longestPalindrome(string s) {
    std::unordered_map<char, int> m;
    for (int i = 0; i < s.size(); i++) {
      m[s[i]]++;
    }

    int even = 0;
    int odd = 0;
    for (auto it : m) {
      int remainder = it.second % 2;
      even += it.second - remainder;
      if (remainder != 0) {
        odd = 1;
      }
    }

    return (even + odd);
  }
};
