class Solution {
public:
  bool isAlienSorted(vector<string>& words, string order) {
    unordered_map<char, int> mp;

    for (int i = 0; i < order.length(); i++) {
      mp[order[i]] = i;
    }

    int n = words.size();
    for (int i = 1; i < n; i++) {
      string first = words[i - 1];
      string second = words[i];

      bool sorted = false;
      for (int j = 0; j < min(first.size(), second.size()); j++) {
        if (mp[first[j]] > mp[second[j]]) {
          return false;
        } else if (mp[first[j]] < mp[second[j]]) {
          sorted = true;
          break;
        } else {
          continue;
        }
      }
      //["apple","app"]
      //"abcdefghijklmnopqrstuvwxyz"
      if (first.size() > second.size() && sorted == false) return false;
    }
    return true;
  }
};
