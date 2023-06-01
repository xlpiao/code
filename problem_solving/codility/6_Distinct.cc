
// you can use includes, for example:
// #include <algorithm>

// you can write to stdout for debugging purposes, e.g.
// cout << "this is a debug message" << endl;

#include <algorithm>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

int solution(vector<int> &A) {
  std::set<int> s;
  for (auto a : A) {
    s.insert(a);
  }
  return s.size();
}

int main(void) {
  vector<int> A{2, 1, 1, 2, 3, 1};
  auto ans = solution(A);
  cout << ans << endl;

  return 0;
}
