class Solution {
public:
  bool canReach(string s, int minJump, int maxJump) {
    int len = s.size();
    int prev = 0;

    queue<int> q;
    q.push(0);

    while (!q.empty()) {
      int i = q.front();
      q.pop();
      if (i == len - 1) return true;

      for (int j = max(i + minJump, prev + 1); j <= min(i + maxJump, len - 1);
           j++) {
        if (s[j] == '0') {
          q.push(j);
        }
        prev = max(prev, i + maxJump);
      }
    }

    return false;
  }
};
