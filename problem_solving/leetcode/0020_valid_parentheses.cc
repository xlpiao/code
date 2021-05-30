#include <iostream>
#include <stack>
#include <string>

class Solution {
public:
  bool isValid(std::string s) {
    std::stack<char> st;

    for (int i = 0; i < s.size(); i++) {
      if (st.empty()) {
        st.push(s[i]);
      } else if ((st.top() == '{' && s[i] == '}') ||
                 (st.top() == '[' && s[i] == ']') ||
                 (st.top() == '(' && s[i] == ')')) {
        st.pop();
      } else {
        st.push(s[i]);
      }
    }

    return st.empty();
  }
};

int main(void) {
  Solution s;

  std::string input{"()"};
  bool ret = s.isValid(input);
  std::cout << ret << std::endl;
}
