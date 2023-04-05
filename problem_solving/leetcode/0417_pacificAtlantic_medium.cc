#include <iostream>
#include <vector>

using namespace std;

class Solution {
 public:
  void dfs(vector<vector<int>>& heights, vector<vector<bool>>& visited, int r,
           int c, int row, int col) {
    visited[r][c] = true;

    if (r - 1 >= 0 && !visited[r - 1][c] &&
        heights[r - 1][c] >= heights[r][c]) {
      dfs(heights, visited, r - 1, c, row, col);
    }
    if (r + 1 < row && !visited[r + 1][c] &&
        heights[r + 1][c] >= heights[r][c]) {
      dfs(heights, visited, r + 1, c, row, col);
    }
    if (c - 1 >= 0 && !visited[r][c - 1] &&
        heights[r][c - 1] >= heights[r][c]) {
      dfs(heights, visited, r, c - 1, row, col);
    }
    if (c + 1 < col && !visited[r][c + 1] &&
        heights[r][c + 1] >= heights[r][c]) {
      dfs(heights, visited, r, c + 1, row, col);
    }
  }

  vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
    int row = heights.size();
    int col = heights[0].size();

    vector<vector<bool>> pacific(row, vector<bool>(col));
    vector<vector<bool>> atlantic(row, vector<bool>(col));

    for (int r = 0; r < row; r++) {
      dfs(heights, pacific, r, 0, row, col);
      dfs(heights, atlantic, r, col - 1, row, col);
    }
    for (int c = 0; c < col; c++) {
      dfs(heights, pacific, 0, c, row, col);
      dfs(heights, atlantic, row - 1, c, row, col);
    }

    vector<vector<int>> result;
    for (int r = 0; r < row; r++) {
      for (int c = 0; c < col; c++) {
        if (pacific[r][c] && atlantic[r][c]) {
          vector<int> pos{r, c};
          result.push_back(pos);
        }
      }
    }
    return result;
  }
};

int main(void) {
  Solution s;

  vector<vector<int>> heights{{1, 2, 2, 3, 5},
                              {3, 2, 3, 4, 4},
                              {2, 4, 5, 3, 1},
                              {6, 7, 1, 4, 5},
                              {5, 1, 1, 2, 4}};

  auto result = s.pacificAtlantic(heights);
  for (auto it : result) {
    cout << "(" << it[0] << "," << it[1] << ")" << endl;
  }

  return 0;
}
