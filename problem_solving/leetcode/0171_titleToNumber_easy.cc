class Solution {
public:
  int titleToNumber(string columnTitle) {
    int col = 0;
    for (int i = 0; i < columnTitle.size(); i++) {
      int idx = columnTitle[i] - 65 + 1;
      col = col * 26 + idx;
    }

    return col;
  }
};
