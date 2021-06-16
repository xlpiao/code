#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  bool isValid(vector<vector<char>>& board, int row, int col) {
    for (int i = 0; i < board.size(); i++) {
      if (i != col && board[row][i] == board[row][col]) return false;
      if (i != row && board[i][col] == board[row][col]) return false;
      int r = 3 * (row / 3) + i / 3;
      int c = 3 * (col / 3) + i % 3;
      if (r != row && c != col && board[r][c] == board[row][col]) return false;
    }
    return true;
  }

  bool isValidSudoku(vector<vector<char>>& board) {
    for (int row = 0; row < board.size(); row++) {
      for (int col = 0; col < board[0].size(); col++) {
        if (board[row][col] != '.' && !isValid(board, row, col)) {
          return false;
        }
      }
    }
    return true;
  }
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
  cout << s.isValidSudoku(board) << endl;
  return 0;
}
