#include <deque>
#include <iostream>
#include <vector>

using namespace std;

class Solution {
 private:
  vector<vector<int>> direction = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

  void print(vector<vector<int>>& grid, int row, int col) {
    for (int r = 0; r < row; r++) {
      for (int c = 0; c < col; c++) {
        cout << grid[r][c] << ", ";
      }
      cout << endl;
    }
    cout << endl;
  }

#if 1
  void dfs(vector<vector<int>>& grid, int r, int c, int row, int col,
           deque<pair<int, int>>& q) {
    if (r < 0 || r >= row || c < 0 || c >= col || grid[r][c] != 1) return;
    q.push_back({r, c});
    grid[r][c] = 2;
    dfs(grid, r - 1, c, row, col, q);
    dfs(grid, r + 1, c, row, col, q);
    dfs(grid, r, c - 1, row, col, q);
    dfs(grid, r, c + 1, row, col, q);
  }
#else
  void dfs(vector<vector<int>>& grid, int x, int y, int row, int col,
           deque<pair<int, int>>& q) {
    q.push_back({x, y});
    grid[x][y] = 2;
    for (int i = 0; i < 4; i++) {
      int r = x + direction[i][0];
      int c = y + direction[i][1];
      if (r >= 0 && c >= 0 && r < row && c < col && grid[r][c] == 1) {
        dfs(grid, r, c, row, col, q);
      }
    }
  }
#endif

 public:
  int shortestBridge(vector<vector<int>>& grid) {
    int row = grid.size();
    int col = grid[0].size();

    print(grid, row, col);

    deque<pair<int, int>> q;
    bool found = false;
    for (int r = 0; r < row; r++) {
      for (int c = 0; c < col; c++) {
        if (grid[r][c] == 1) {
          dfs(grid, r, c, row, col, q);
          found = true;
          break;
        }
      }
      if (found) {
        break;
      }
    }

    print(grid, row, col);

    int dist = 0;
    while (!q.empty()) {
      int len = q.size();
      while (len-- > 0) {
        pair<int, int> currPos = q.front();
        q.pop_front();

        int x = currPos.first;
        int y = currPos.second;

        for (int i = 0; i < 4; i++) {
          int r = x + direction[i][0];
          int c = y + direction[i][1];
          if (r >= 0 && c >= 0 && r < row && c < col) {
            if (grid[r][c] == 1) {
              return dist;
            }
            if (grid[r][c] == 0) {
              grid[r][c] = 3;
              q.push_back({r, c});
            }
          }
        }
      }
      dist++;
      cout << "dist = " << dist << endl;
      print(grid, row, col);
    }
    return -1;
  }
};

int main(void) {
  Solution s;
  vector<vector<int>> grid{{1, 0, 0, 0, 1},
                           {1, 0, 0, 0, 1},
                           {1, 0, 0, 1, 1},
                           {1, 0, 0, 0, 1},
                           {1, 0, 0, 1, 1}};

  s.shortestBridge(grid);

  return 0;
}
