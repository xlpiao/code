#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

using namespace std;

int solution(vector<int> &A) {
  int n = A.size();
  std::unordered_map<int, int> mp;

  for (int i = 0; i < n; ++i) {
    mp[A[i]]++;
    if (mp[A[i]] > n / 2) return i;
  }

  return -1;
}

int main(void) {
  vector<int> A{3, 4, 3, 2, 3, -1, 3, 3};
  auto ans = solution(A);
  cout << ans << endl;

  return 0;
}
