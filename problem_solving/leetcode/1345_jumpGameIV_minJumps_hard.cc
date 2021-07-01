#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

using namespace std;

class Solution {
public:
  struct pos {
    int i;
    int step;
    pos(int i, int step) : i(i), step(step){};
  };

  int minJumps(vector<int>& arr) {
    auto n = arr.size();

    if (n <= 1) return 0;

    vector<int> visited(n, 0);
    queue<pos> q;

    q.push(pos(0, 0));
    visited[0] = 1;

    unordered_map<int, vector<int>> mp;

    for (int i = 0; i < n; i++) {
      if (i + 1 < n && i - 1 >= 0 && arr[i - 1] == arr[i] &&
          arr[i] == arr[i + 1])
        continue;

      mp[arr[i]].push_back(i);
    }

    int ret = 0;
    while (!q.empty()) {
      auto curr = q.front();
      q.pop();

      if (curr.i == n - 1) {
        ret = curr.step;
        break;
      }

      if (curr.i - 1 >= 0 && !visited[curr.i - 1]) {
        q.push(pos(curr.i - 1, curr.step + 1));
        visited[curr.i - 1] = 1;
      }

      if (curr.i + 1 < n && !visited[curr.i + 1]) {
        q.push(pos(curr.i + 1, curr.step + 1));
        visited[curr.i + 1] = 1;
      }

      for (auto i : mp[arr[curr.i]]) {
        if (i != curr.i && !visited[i]) {
          q.push(pos(i, curr.step + 1));
          visited[i] = 1;
        }
      }
    }

    return ret;
  }
};

int main(void) {
  Solution s;
  int minStep;

  vector<int> arr{100, -23, -23, 404, 100, 23, 23, 23, 3, 404};
  minStep = s.minJumps(arr);
  cout << minStep << endl;

  vector<int> arr1(10, 7);
  arr1[9] = 10;
  minStep = s.minJumps(arr1);
  cout << minStep << endl;

  vector<int> arr2(10, 7);
  minStep = s.minJumps(arr2);
  cout << minStep << endl;

  return 0;
}
