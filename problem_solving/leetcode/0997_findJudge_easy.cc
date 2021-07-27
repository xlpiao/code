class Solution {
public:
  int findJudge(int n, vector<vector<int>>& trust) {
    vector<vector<int>> followers(n + 1), following(n + 1);

    for (auto x : trust) {
      followers[x[1]].push_back(x[0]);
      following[x[0]].push_back(x[1]);
    }

    for (int i = 1; i <= n; i++) {
      if (followers[i].size() == n - 1 && following[i].size() == 0) {
        return i;
      }
    }

    return -1;
  }
};
