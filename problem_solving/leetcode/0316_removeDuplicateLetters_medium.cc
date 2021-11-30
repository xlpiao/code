class Solution {
public:
  string removeDuplicateLetters(string s) {
    unordered_map<char, pair<int, bool> > m;
    string output;

    for (int i = 0; i < s.size(); i++) m[s[i]].first++;

    for (int i = 0; i < s.size(); i++) {
      m[s[i]].first--;

      if (m[s[i]].second) continue;

      while (s[i] < output.back() && m[output.back()].first >= 1) {
        m[output.back()].second = false;
        output.pop_back();
      }

      output += s[i];
      m[s[i]].second = true;
    }

    return output;
  }
};
