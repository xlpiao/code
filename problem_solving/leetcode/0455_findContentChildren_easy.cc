class Solution {
public:
  int findContentChildren(vector<int>& g, vector<int>& s) {
    int content = 0;
    int j = 0;
    sort(g.begin(), g.end());
    sort(s.begin(), s.end());
    for (int i = 0; i < g.size(); i++) {
      while (j < s.size()) {
        if (g[i] <= s[j]) {
          content++;
          j++;
          break;
        } else {
          j++;
        }
      }
    }
    return content;
  }
};

class Solution {
public:
  int findContentChildren(vector<int>& g, vector<int>& s) {
    sort(g.begin(), g.end());
    sort(s.begin(), s.end());
    int i = 0, j = 0;
    while (i < g.size() && j < s.size()) {
      if (g[i] <= s[j]) ++i;
      ++j;
    }
    return i;
  }
};
