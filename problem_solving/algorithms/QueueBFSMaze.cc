/**
 * File              : queue_bfs_maze.cpp
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2018.08.11
 * Last Modified Date: 2018.08.12
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */
#include <iostream>
#include <queue>
using namespace std;

char maze[6][6];
int visited[6][6];
char line[6];

struct POS {
  int x;
  int y;
  int len;
};

void print_queue(std::queue<POS> q) {
  while (!q.empty()) {
    cout << q.front().x << " " << q.front().y << " " << q.front().len
         << endl;
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
    cout << line << endl;
  }

  queue<POS> path;
  int x = 0, y = 0, len = 1;
  POS begin = {x, y, len};

  path.push(begin);
  visited[0][0] = 1;

  int shortest = 0;

  while (!path.empty()) {
    cout << "Debug: -----Start-----" << endl;
    cout << "Visit [" << x << ", " << y << "], "
         << "len: " << len << endl;
    x = path.front().x;
    y = path.front().y;
    len = path.front().len;
    path.pop();

    if (x == row - 1 && y == col - 1) {
      shortest = len;
      break;
    }

    /* up*/
    if (x - 1 >= 0 && maze[x - 1][y] == '1' && visited[x - 1][y] == 0) {
      POS up = {x - 1, y, len + 1};
      path.push(up);
      visited[x - 1][y] = 1;
      cout << "top" << endl;
    }
    /* down */
    if (x + 1 < row && maze[x + 1][y] == '1' && visited[x + 1][y] == 0) {
      POS down = {x + 1, y, len + 1};
      path.push(down);
      visited[x + 1][y] = 1;
      cout << "down" << endl;
    }
    /* left */
    if (y - 1 >= 0 && maze[x][y - 1] == '1' && visited[x][y - 1] == 0) {
      POS left = {x, y - 1, len + 1};
      path.push(left);
      visited[x][y - 1] = 1;
      cout << "left" << endl;
    }
    /* right*/
    if (y + 1 < col && maze[x][y + 1] == '1' && visited[x][y + 1] == 0) {
      POS right = {x, y + 1, len + 1};
      path.push(right);
      visited[x][y + 1] = 1;
      cout << "right" << endl;
    }
    print_queue(path);
    cout << "Debug: -----End-----" << endl;
  }

  cout << "The shortest path: " << shortest << endl;

  return 0;
}
