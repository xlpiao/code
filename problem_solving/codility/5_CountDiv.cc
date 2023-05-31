#include <iostream>
#include <vector>
using namespace std;

int solution(int A, int B, int K) {
  int cnt = B / K - A / K;
  if (A % K == 0) ++cnt;

  return cnt;
}

int main(void) {
  auto ans = solution(5, 11, 2);
  cout << ans << endl;

  return 0;
}
