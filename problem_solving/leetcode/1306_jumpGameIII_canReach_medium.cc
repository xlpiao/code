class Solution {
public:
  //// BFS
  bool canReach(vector<int>& arr, int start) {
    int len = arr.size();
    vector<bool> visited(len, false);

    queue<int> q;
    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
      int idx = q.front();
      q.pop();

      if (arr[idx] == 0) {
        return true;
      }

      int left = idx - arr[idx];
      if (left >= 0 && !visited[left]) {
        q.push(left);
        visited[left] = true;
      }

      int right = idx + arr[idx];
      if (right < len && !visited[right]) {
        q.push(right);
        visited[right] = true;
      }
    }
    return false;
  }
};

#if 0
class Solution {
public:
  bool canReach(vector<int>& arr, int start) {
    if (start < 0 || start >= arr.size()) return false;
    if (arr[start] == 0) return true;

    dp[start] = 1;

    bool leftBranch = 0;
    bool rightBranch = 0;

    int left = start - arr[start];
    if (left >= 0 && !dp[left]) {
      leftBranch = canReach(arr, left);
    }

    int right = start + arr[start];
    if (right < arr.size() && !dp[right]) {
      rightBranch = canReach(arr, right);
    }

    return leftBranch || rightBranch;
  }

private:
  unordered_map<int, int> dp;
};
#endif
