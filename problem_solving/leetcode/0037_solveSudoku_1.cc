#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  bool isValid(vector<vector<char>>& board, int row, int col, char ch) {
    for (int i = 0; i < board.size(); i++) {
      if (i != col && board[row][i] == ch) return false;
      if (i != row && board[i][col] == ch) return false;
      int r = 3 * (row / 3) + i / 3;
      int c = 3 * (col / 3) + i % 3;
      if (r != row && c != col && board[r][c] == ch) return false;
    }
    return true;
  }

  bool dfs(vector<vector<char>>& board, int row, int col) {
    if (row >= board.size()) return true;

    if (board[row][col] == '.') {
      for (char ch = '1'; ch <= '9'; ch++) {
        if (isValid(board, row, col, ch)) {
          board[row][col] = ch;
          int nextCol = (col + 1) % board[0].size();
          int nextRow = row + (col == 8);
          if (dfs(board, nextRow, nextCol)) return true;
          board[row][col] = '.';
        }
      }
      return false;
    } else {
      int nextCol = (col + 1) % board[0].size();
      int nextRow = row + (col == 8);
      return dfs(board, nextRow, nextCol);
    }
    return true;
  }

  void solveSudoku(vector<vector<char>>& board) { dfs(board, 0, 0); }
};

int main(void) {
  Solution s;

  vector<vector<char>> board{{'5', '3', '.', '.', '7', '.', '.', '.', '.'},
                             {'6', '.', '.', '1', '9', '5', '.', '.', '.'},
                             {'.', '9', '8', '.', '.', '.', '.', '6', '.'},
                             {'8', '.', '.', '.', '6', '.', '.', '.', '3'},
                             {'4', '.', '.', '8', '.', '3', '.', '.', '1'},
                             {'7', '.', '.', '.', '2', '.', '.', '.', '6'},
                             {'.', '6', '.', '.', '.', '.', '2', '8', '.'},
                             {'.', '.', '.', '4', '1', '9', '.', '.', '5'},
                             {'.', '.', '.', '.', '8', '.', '.', '7', '9'}};
  s.solveSudoku(board);
  return 0;
}
