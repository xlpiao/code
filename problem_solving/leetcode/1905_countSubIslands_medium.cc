#include <iostream>
#include <vector>

using namespace std;

class Solution {
 public:
  void dfs(vector<vector<int>>& grid1, vector<vector<int>>& grid2, int r, int c,
           int row, int col, int& check) {
    if (r < 0 || c < 0 || r >= row || c >= col || grid2[r][c] == 0) return;

    if (grid1[r][c] != grid2[r][c]) {
      check = 0;
      return;
    }

    grid1[r][c] = 0;
    grid2[r][c] = 0;

    dfs(grid1, grid2, r + 1, c, row, col, check);
    dfs(grid1, grid2, r, c + 1, row, col, check);
    dfs(grid1, grid2, r - 1, c, row, col, check);
    dfs(grid1, grid2, r, c - 1, row, col, check);
  }

  int countSubIslands(vector<vector<int>>& grid1, vector<vector<int>>& grid2) {
    int row = grid1.size();
    int col = grid1[0].size();
    int ans = 0;

    for (int r = 0; r < row; r++) {
      for (int c = 0; c < col; c++) {
        if (grid2[r][c] == 1) {
          int check = 1;
          dfs(grid1, grid2, r, c, row, col, check);
          ans += check;
        }
      }
    }
    return ans;
  }
};

int main(void) {
  vector<vector<int>> grid1{{1, 1, 1, 0, 0},
                            {0, 1, 1, 1, 1},
                            {0, 0, 0, 0, 0},
                            {1, 0, 0, 0, 0},
                            {1, 1, 0, 1, 1}};

  vector<vector<int>> grid2{{1, 1, 1, 0, 0},
                            {0, 0, 1, 1, 1},
                            {0, 1, 0, 0, 0},
                            {1, 0, 1, 1, 0},
                            {0, 1, 0, 1, 0}};

  Solution s;
  auto ans = s.countSubIslands(grid1, grid2);
  cout << ans << endl;

  return 0;
}
