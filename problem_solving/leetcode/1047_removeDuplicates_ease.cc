using namespace std;

class Solution {
public:
  string removeDuplicates(string s) {
    if (s.empty()) return s;

    string st;
    int len = s.size();
    for (int i = 0; i < len; i++) {
      if (st.empty() || st.back() != s[i]) {
        st.push_back(s[i]);
      } else {
        st.pop_back();
      }
    }
    return st;
  }
};
