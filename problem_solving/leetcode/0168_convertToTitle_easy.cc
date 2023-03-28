#include <iostream>
#include <string>

using namespace std;

class Solution {
public:
  string convertToTitle(int columnNumber) {
    std::string result{};

    while (columnNumber > 0) {
      columnNumber -= 1;
      result.insert(result.begin(), columnNumber % 26 + 65);
      columnNumber /= 26;
    }

    return result;
  }
};

int main(void) {
  Solution S;
  string ans = S.convertToTitle(27);

  cout << ans << endl;

  return 0;
}
