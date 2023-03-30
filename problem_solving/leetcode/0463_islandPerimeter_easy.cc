class Solution {
public:
  int islandPerimeter(vector<vector<int>>& grid) {
    int ans = 0;

    int row = grid.size();
    int col = grid[0].size();

    for (int r = 0; r < row; r++) {
      for (int c = 0; c < col; c++) {
        if (grid[r][c] == 1) {
          if (r - 1 < 0 || !grid[r - 1][c]) ans++;
          if (r + 1 > row - 1 || !grid[r + 1][c]) ans++;
          if (c - 1 < 0 || !grid[r][c - 1]) ans++;
          if (c + 1 > col - 1 || !grid[r][c + 1]) ans++;
        }
      }
    }
    return ans;
  }
};
