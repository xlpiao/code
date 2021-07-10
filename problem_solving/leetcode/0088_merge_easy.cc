#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
    int i = m - 1, j = n - 1, k = m + n - 1;
    while (i >= 0 && j >= 0) {
      cout << "[i, j, k] = " << i << ", " << j << ", " << k << endl;
      if (nums1[i] > nums2[j])
        nums1[k--] = nums1[i--];
      else
        nums1[k--] = nums2[j--];

      for (auto it : nums1) {
        cout << it << ", ";
      }
      cout << endl;
    }
    while (j >= 0) nums1[k--] = nums2[j--];
  }
};

int main(void) {
  Solution s;

  vector<int> nums1{1, 2, 3, 0, 0, 0};
  int m = 3;
  vector<int> nums2{2, 5, 6};
  int n = 3;
  s.merge(nums1, m, nums2, n);

  for (auto it : nums1) {
    cout << it << ", ";
  }
  cout << endl;

  return 0;
}
