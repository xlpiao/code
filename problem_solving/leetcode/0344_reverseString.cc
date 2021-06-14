#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  void reverseString(vector<char>& s) {
    int left = 0;
    int right = s.size() - 1;

    while (left < right) {
      swap(s[left++], s[right--]);
    }
  }
};

int main(void) {
  Solution s;

  vector<char> input{'h', 'e', 'l', 'l', 'o'};
  s.reverseString(input);
}
