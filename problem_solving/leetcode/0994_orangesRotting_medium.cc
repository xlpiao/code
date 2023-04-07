class Solution {
 public:
  bool isValid(int r, int c, int row, int col) {
    if (r >= 0 && r < row && c >= 0 && c < col) return true;
    return false;
  }
  int orangesRotting(vector<vector<int>>& grid) {
    vector<int> r_offset = {-1, 1, 0, 0};
    vector<int> c_offset = {0, 0, 1, -1};
    int row = grid.size();
    int col = grid[0].size();

    deque<pair<int, int>> q;
    int fresh = 0;
    for (int r = 0; r < row; r++) {
      for (int c = 0; c < col; c++) {
        if (grid[r][c] == 2) q.push_back({r, c});
        if (grid[r][c] == 1) fresh++;
      }
    }

    int ans = -1;
    while (!q.empty()) {
      int sz = q.size();
      while (sz--) {
        pair<int, int> pos = q.front();
        q.pop_front();
        for (int i = 0; i < 4; i++) {
          int r = pos.first + r_offset[i];
          int c = pos.second + c_offset[i];
          if (isValid(r, c, row, col) && grid[r][c] == 1) {
            grid[r][c] = 2;
            q.push_back({r, c});
            fresh--;
          }
        }
      }
      ans++;
    }
    if (fresh > 0) return -1;
    if (ans == -1) return 0;
    return ans;
  }
};
