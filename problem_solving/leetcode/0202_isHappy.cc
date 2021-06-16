#include <iostream>
#include <unordered_set>

using namespace std;

class Solution {
public:
  bool isHappy(int n) {
    std::unordered_set<int> s;
    while (1) {
      int num = 0;
      while (n > 0) {
        num += (n % 10) * (n % 10);
        n = n / 10;
      }
      if (num == 1) {
        return true;
      } else if (s.find(num) != s.end()) {
        return false;
      } else {
        s.insert(num);
        n = num;
      }
    }
  }
};

int main(void) {
  Solution s;
  cout << s.isHappy(19) << endl;
  return 0;
}
