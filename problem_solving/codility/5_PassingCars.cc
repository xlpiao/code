#include <iostream>
#include <vector>
using namespace std;

int solution(vector<int> &A) {
  int n = A.size();

  int passedCarsToEast = 0;
  int passedCars = 0;
  for (int i = 0; i < n; i++) {
    if (passedCars >= 1000000000) return -1;

    if (A[i] == 0) ++passedCarsToEast;
    if (A[i] == 1) passedCars += passedCarsToEast;
  }
  return passedCars;
}

int main(void) {
  vector<int> A{0, 1, 0, 1, 1};

  auto ans = solution(A);
  cout << ans << endl;

  return 0;
}
