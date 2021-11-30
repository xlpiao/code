class Solution {
public:
  string reverseParentheses(string s) {
    if (s.empty()) return s;

    deque<int> st;

    for (int i = 0; i < s.size(); i++) {
      if (s[i] == '(') {
        st.push_back(i);
      } else if (s[i] == ')') {
        int t = st.back();
        st.pop_back();
        reverse(s.begin() + t + 1, s.begin() + i);
      }
    }

    string ans;
    for (auto it : s) {
      if (it == ')' or it == '(') continue;
      ans += it;
    }

    return ans;
  }
};
