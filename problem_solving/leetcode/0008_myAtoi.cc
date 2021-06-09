#include <iostream>
#include <string>

using namespace std;

class Solution {
public:
  int myAtoi(string s) {
    bool neg = false;
    int i = 0;

    while (s[i] == ' ') {
      i++;
    }
    if (s[i] == '-') {
      i++;
      neg = true;
    } else if (s[i] == '+') {
      i++;
    }

    int num = 0;
    while (isDigit(s[i]) && i < s.size()) {
      if (neg) {
        if (num > INT_MAX / 10 || (num == INT_MAX / 10 && s[i] - '0' >= 8)) {
          return INT_MIN;
        }
      } else {
        if (num > INT_MAX / 10 || (num == INT_MAX / 10 && s[i] - '0' >= 7)) {
          return INT_MAX;
        }
      }

      num = num * 10 + (s[i] - '0');
      i++;
    }
    if (neg) {
      num = (-1) * num;
    }
    return num;
  }

private:
  bool isDigit(char c) {
    if (c >= '0' && c <= '9') {
      return true;
    }
    return false;
  }
};

int main(void) {
  Solution s;

  string input = "   -42";
  int num = s.myAtoi(input);
  cout << num << endl;
}
