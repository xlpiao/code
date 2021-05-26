#include <iostream>
#include <unordered_map>

class Solution {
public:
  int romanToInt(std::string s) {
    std::unordered_map<char, int> mapper{{'I', 1},
                                         {'V', 5},
                                         {'X', 10},
                                         {'L', 50},
                                         {'C', 100},
                                         {'D', 500},
                                         {'M', 1000}};

    int sum = 0;
    for (int i = 0; i < s.size(); i++) {
      if (mapper[s[i]] >= mapper[s[i + 1]]) {
        sum += mapper[s[i]];
      } else {
        sum -= mapper[s[i]];
      }
    }
    return sum;
  }
};

int main(void) {
  Solution s;

  int number = s.romanToInt("IV");
  std::cout << number << std::endl;

  number = s.romanToInt("VI");
  std::cout << number << std::endl;

  number = s.romanToInt("III");
  std::cout << number << std::endl;
}
