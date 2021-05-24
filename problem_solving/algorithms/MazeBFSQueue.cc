/**
 * File              : QueueBFSMaze.cc
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2018.08.11
 * Last Modified Date: 2021.05.23
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */
#include <iostream>
#include <queue>

char maze[6][6];
int visited[6][6];
char line[6];

struct POS {
  int x;
  int y;
  int len;
};

void print(std::queue<POS> q) {
  while (!q.empty()) {
    std::cout << q.front().x << " " << q.front().y << " " << q.front().len
              << std::endl;
    q.pop();
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

  std::queue<POS> q;
  int x = 0, y = 0, len = 1;
  POS begin = {x, y, len};

  q.push(begin);
  visited[0][0] = 1;

  int shortest = 0;

  while (!q.empty()) {
    std::cout << "Debug: -----Start-----" << std::endl;
    std::cout << "Visit [" << x << ", " << y << "], "
              << "len: " << len << std::endl;
    x = q.front().x;
    y = q.front().y;
    len = q.front().len;
    q.pop();

    if (x == row - 1 && y == col - 1) {
      shortest = len;
      break;
    }

    /* up*/
    if (x - 1 >= 0 && maze[x - 1][y] == '1' && visited[x - 1][y] == 0) {
      POS up = {x - 1, y, len + 1};
      q.push(up);
      visited[x - 1][y] = 1;
      std::cout << "up" << std::endl;
    }
    /* down */
    if (x + 1 < row && maze[x + 1][y] == '1' && visited[x + 1][y] == 0) {
      POS down = {x + 1, y, len + 1};
      q.push(down);
      visited[x + 1][y] = 1;
      std::cout << "down" << std::endl;
    }
    /* left */
    if (y - 1 >= 0 && maze[x][y - 1] == '1' && visited[x][y - 1] == 0) {
      POS left = {x, y - 1, len + 1};
      q.push(left);
      visited[x][y - 1] = 1;
      std::cout << "left" << std::endl;
    }
    /* right*/
    if (y + 1 < col && maze[x][y + 1] == '1' && visited[x][y + 1] == 0) {
      POS right = {x, y + 1, len + 1};
      q.push(right);
      visited[x][y + 1] = 1;
      std::cout << "right" << std::endl;
    }
    print(q);
    std::cout << "Debug: -----End-----" << std::endl;
  }

  std::cout << "The shortest path: " << shortest << std::endl;

  return 0;
}
