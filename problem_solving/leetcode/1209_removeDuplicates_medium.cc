using namespace std;

class Solution {
public:
  string removeDuplicates(string s, int k) {
    if (s.empty()) return s;

    deque<pair<int, char>> st;

    int len = s.size();
    for (int i = 0; i < len; i++) {
      if (st.empty() || s[i] != st.back().second) {
        st.push_back({1, s[i]});
      } else {
        st.back().first++;
        if (st.back().first == k) {
          st.pop_back();
        }
      }
    }

    string ret;
    for (auto it : st) {
      ret += string(it.first, it.second);
    }

    return ret;
  }
};
