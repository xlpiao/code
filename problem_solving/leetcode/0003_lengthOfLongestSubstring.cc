#include <iostream>
#include <string>
#include <unordered_map>

class Solution {
public:
  int lengthOfLongestSubstring(std::string s) {
    std::unordered_map<int, int> mp;
    int substrStart = 0;
    int maxlen = 0;
    for (int i = 0; i < s.size(); i++) {
      mp[s[i]]++;
      if (mp.size() == i - substrStart + 1) {
        if (maxlen < mp.size()) {
          maxlen = mp.size();
        }
      } else if (mp.size() < i - substrStart + 1) {
        mp[s[substrStart]]--;
        if (mp[s[substrStart]] == 0) {
          mp.erase(s[substrStart]);
        }
        substrStart++;
      }
    }
    return maxlen;
  }
};
