#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  int trap(vector<int>& height) {
    int n = height.size();
    if (n == 0) return 0;

    vector<int> maxLeft(n, 0);
    vector<int> maxRight(n, 0);

    maxLeft[0] = height[0];
    for (int i = 1; i < n; i++) {
      maxLeft[i] = max(maxLeft[i - 1], height[i]);
    }

    maxRight[n - 1] = height[n - 1];
    for (int i = n - 2; i >= 0; i--) {
      maxRight[i] = max(maxRight[i + 1], height[i]);
    }

    int sum = 0;
    for (int i = 1; i < n; i++) {
      int m = min(maxLeft[i], maxRight[i]);
      sum += (m - height[i]) > 0 ? (m - height[i]) : 0;
    }
    return sum;
  }
};

int main(void) {
  Solution s;

  vector<int> height{4, 2, 0, 3, 2, 5};
  int res = s.trap(height);
  cout << res << endl;

  return 0;
}
