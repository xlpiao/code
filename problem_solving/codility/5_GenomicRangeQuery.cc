
// you can use includes, for example:
// #include <algorithm>

// you can write to stdout for debugging purposes, e.g.
// cout << "this is a debug message" << endl;

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

vector<int> solution(string &S, vector<int> &P, vector<int> &Q) {
  vector<int> res;
  int N = S.size();
  vector<int> A(N + 1, 0);
  vector<int> C(N + 1, 0);
  vector<int> G(N + 1, 0);
  vector<int> T(N + 1, 0);

  for (int i = 0; i < N; i++) {
    A[i + 1] = A[i] + (S[i] == 'A' ? 1 : 0);
    C[i + 1] = C[i] + (S[i] == 'C' ? 1 : 0);
    G[i + 1] = G[i] + (S[i] == 'G' ? 1 : 0);
    T[i + 1] = T[i] + (S[i] == 'T' ? 1 : 0);
  }
  for (auto it : A) {
    cout << it << ", ";
  }
  for (size_t i = 0; i < P.size(); i++) {
    int start = P[i];
    int end = Q[i];
    cout << start << "->" << end << endl;
    if (A[end + 1] - A[start] > 0)
      res.push_back(1);
    else if (C[end + 1] - C[start] > 0)
      res.push_back(2);
    else if (G[end + 1] - G[start] > 0)
      res.push_back(3);
    else if (T[end + 1] - T[start] > 0)
      res.push_back(4);
  }
  return res;
}

int main(void) {
  string S{"CAGCCTA"};
  vector<int> P{2, 5, 0};
  vector<int> Q{4, 5, 6};

  auto ans = solution(S, P, Q);
  for (auto it : ans) {
    cout << it << ", ";
  }

  return 0;
}
