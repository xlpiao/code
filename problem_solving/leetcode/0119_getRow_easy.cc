#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  vector<int> getRow(int rowIndex) {
    auto numRows = rowIndex + 1;
    vector<vector<int>> num(numRows);

    for (int i = 0; i < numRows; i++) {
      num[i].resize(i + 1);
      num[i][0] = 1;
      num[i][i] = 1;

      for (int j = 1; j < i; j++) {
        num[i][j] = num[i - 1][j - 1] + num[i - 1][j];
      }
    }
    return num[rowIndex];
  }
};

int main(void) {
  Solution S;
  auto ans = S.getRow(3);

  for (auto it : ans) {
    std::cout << it << std::endl;
  }
  return 0;
}
