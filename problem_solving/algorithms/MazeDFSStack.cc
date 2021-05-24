/**
 * File              : StackDFSMaze.cc
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2018.08.11
 * Last Modified Date: 2021.05.23
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */
#include <iostream>
#include <stack>

char maze[6][6];
int visited[6][6];
char line[6];

struct POS {
  int x;
  int y;
  int len;
};

void print(std::stack<POS> s) {
  while (!s.empty()) {
    std::cout << s.top().x << " " << s.top().y << " " << s.top().len
              << std::endl;
    s.pop();
  }
}

int main(void) {
  int row, col;
  scanf("%d%d", &row, &col);
  for (int i = 0; i < row; i++) {
    scanf("%s", line);
    for (int j = 0; j < col; j++) {
      maze[i][j] = line[j];
      visited[i][j] = 0;
    }
    std::cout << line << std::endl;
  }

  std::stack<POS> s;
  int x = 0, y = 0, len = 1;
  POS begin = {x, y, len};

  s.push(begin);
  visited[0][0] = 1;

  int shortest = 0;

  while (!s.empty()) {
    std::cout << "Debug: -----Start-----" << std::endl;
    std::cout << "Visit [" << x << ", " << y << "], "
              << "len: " << len << std::endl;
    x = s.top().x;
    y = s.top().y;
    len = s.top().len;
    s.pop();

    if (x == row - 1 && y == col - 1) {
      shortest = len;
      break;
    }

    /* up*/
    if (x - 1 >= 0 && maze[x - 1][y] == '1' && visited[x - 1][y] == 0) {
      POS up = {x - 1, y, len + 1};
      s.push(up);
      visited[x - 1][y] = 1;
      std::cout << "up" << std::endl;
    }
    /* down */
    if (x + 1 < row && maze[x + 1][y] == '1' && visited[x + 1][y] == 0) {
      POS down = {x + 1, y, len + 1};
      s.push(down);
      visited[x + 1][y] = 1;
      std::cout << "down" << std::endl;
    }
    /* left */
    if (y - 1 >= 0 && maze[x][y - 1] == '1' && visited[x][y - 1] == 0) {
      POS left = {x, y - 1, len + 1};
      s.push(left);
      visited[x][y - 1] = 1;
      std::cout << "left" << std::endl;
    }
    /* right*/
    if (y + 1 < col && maze[x][y + 1] == '1' && visited[x][y + 1] == 0) {
      POS right = {x, y + 1, len + 1};
      s.push(right);
      visited[x][y + 1] = 1;
      std::cout << "right" << std::endl;
    }
    print(s);
    std::cout << "Debug: -----End-----" << std::endl;
  }

  std::cout << "The shortest s: " << shortest << std::endl;

  return 0;
}
