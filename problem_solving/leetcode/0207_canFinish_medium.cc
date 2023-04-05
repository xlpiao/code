class Solution {
 public:
  bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    unordered_map<int, vector<int>> m;
    for (int i = 0; i < prerequisites.size(); i++) {
      int u = prerequisites[i][0];
      int v = prerequisites[i][1];
      m[u].push_back(v);
    }

    unordered_set<int> visited;
    for (int course = 0; course < numCourses; course++) {
      if (!dfs(course, m, visited)) {
        return false;
      }
    }
    return true;
  }

 private:
  bool dfs(int course, unordered_map<int, vector<int>>& m,
           unordered_set<int>& visited) {
    if (visited.count(course) != 0) {
      return false;
    }
    if (m[course].empty()) {
      return true;
    }
    visited.insert(course);
    for (int i = 0; i < m[course].size(); i++) {
      int nextCourse = m[course][i];
      if (!dfs(nextCourse, m, visited)) {
        return false;
      }
    }
    m[course].clear();
    visited.erase(course);
    return true;
  }
};
