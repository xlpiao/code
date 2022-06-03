/**
 * File              : 1_two_sum.cc
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2021.05.23
 * Last Modified Date: 2021.05.23
 * Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>
 */

#include <iostream>
#include <unordered_map>
#include <vector>

class Solution {
public:
  int uniqueMorseRepresentations(std::vector<std::string>& words) {
    std::vector<std::string> code{
        ".-",   "-...", "-.-.", "-..",  ".",   "..-.", "--.",  "....", "..",
        ".---", "-.-",  ".-..", "--",   "-.",  "---",  ".--.", "--.-", ".-.",
        "...",  "-",    "..-",  "...-", ".--", "-..-", "-.--", "--.."};

    std::unordered_map<std::string, int> morseRep;
    for (int i = 0; i < words.size(); i++) {
      std::string encode;
      for (int j = 0; j < words[i].size(); j++) {
        int idx = words[i][j] - 'a';
        encode.append(code[idx]);
      }
      morseRep.emplace(encode, 0);
    }

    return morseRep.size();
  }
};

int main(void) {
  Solution s;
  std::vector<std::string> words{"gin", "zen", "gig", "msg"};

  auto result = s.uniqueMorseRepresentations(words);
  std::cout << result << std::endl;

  return 0;
}
