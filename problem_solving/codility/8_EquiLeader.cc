#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

using namespace std;

int solution(vector<int> &A) {
  int n = A.size();

  int leader = -1;
  int leader_count = 0;
  std::unordered_map<int, int> mp;
  for (int i = 0; i < n; ++i) {
    mp[A[i]]++;
    if (mp[A[i]] > n / 2) {
      leader = A[i];
      leader_count = mp[A[i]];
    }
  }

  int equiLeaders = 0;
  int leaders = 0;
  for (int i = 0; i < n; i++) {
    if (A[i] == leader) {
      ++leaders;
    }
    if (leaders > (i + 1) / 2 && leader_count - leaders > (n - 1 - i) / 2) {
      ++equiLeaders;
    }
  }

  return equiLeaders;
}

int main(void) {
  vector<int> A{3, 4, 3, 2, 3, -1, 3, 3};
  auto ans = solution(A);
  cout << ans << endl;

  return 0;
}
