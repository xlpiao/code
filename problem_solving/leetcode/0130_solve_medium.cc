#include <iostream>
#include <vector>

using namespace std;

class Solution {
 public:
  void print(vector<vector<char>>& board) {
    for (auto it : board) {
      for (auto elem : it) {
        cout << elem << ", ";
      }
      cout << endl;
    }
  }

  void solve(vector<vector<char>>& board) {
    int m = board.size();
    int n = board[0].size();

    for (int r = 0; r < m; r++) {
      dfs(board, r, 0, m, n);
      dfs(board, r, n - 1, m, n);
    }

    for (int c = 0; c < n; c++) {
      dfs(board, 0, c, m, n);
      dfs(board, m - 1, c, m, n);
    }

    print(board);

    for (int r = 0; r < m; r++) {
      for (int c = 0; c < n; c++) {
        if (board[r][c] == 'O') {
          board[r][c] = 'X';
        }
        if (board[r][c] == 'E') {
          board[r][c] = 'O';
        }
      }
    }
  }

 private:
  void dfs(vector<vector<char>>& board, int r, int c, int m, int n) {
    if (r < 0 || r >= m || c < 0 || c >= n || board[r][c] != 'O') {
      // cout << r << ", " << c << endl;
      return;
    }
    // cout << r << ", " << c << endl;
    board[r][c] = 'E';

    dfs(board, r - 1, c, m, n);
    dfs(board, r + 1, c, m, n);
    dfs(board, r, c - 1, m, n);
    dfs(board, r, c + 1, m, n);
  }
};

int main(void) {
  Solution s;
  vector<vector<char>> board{{'X', 'X', 'X', 'X'},
                             {'X', 'O', 'O', 'X'},
                             {'X', 'X', 'O', 'X'},
                             {'X', 'O', 'X', 'X'}};
  s.solve(board);

  return 0;
}
