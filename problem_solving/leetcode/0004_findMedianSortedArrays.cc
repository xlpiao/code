#include <iostream>
#include <vector>

class Solution {
public:
  double findMedianSortedArrays(std::vector<int>& nums1,
                                std::vector<int>& nums2) {
    int middle;
    double median = 0.0;
    if (nums1.empty()) {
      middle = nums2.size() / 2;
      if (nums2.size() % 2 == 0) {
        median = ((double)nums2[middle - 1] + (double)nums2[middle]) / 2.0;
      } else {
        median = (double)nums2[middle];
      }
      return median;
    }

    if (nums2.empty()) {
      middle = nums1.size() / 2;
      if (nums1.size() % 2 == 0) {
        median = ((double)nums1[middle - 1] + (double)nums1[middle]) / 2.0;
      } else {
        median = (double)nums1[middle];
      }
      return median;
    }

    middle = (nums1.size() + nums2.size()) / 2;
    int size = middle + 1;

    std::vector<int> num;
    int idx1 = 0, idx2 = 0;
    while (num.size() < size) {
      if (idx2 < nums2.size() && idx1 == nums1.size()) {
        num.push_back(nums2[idx2++]);
      } else if (idx1 < nums1.size() && idx2 == nums2.size()) {
        num.push_back(nums1[idx1++]);
      } else if (nums1[idx1] <= nums2[idx2]) {
        num.push_back(nums1[idx1++]);
      } else if (nums1[idx1] > nums2[idx2]) {
        num.push_back(nums2[idx2++]);
      }
    }

    if ((nums1.size() + nums2.size()) % 2 != 0) {
      median = (double)num[middle];
    } else {
      median = ((double)num[middle - 1] + (double)num[middle]) / 2;
    }
    return median;
  }
};

int main(void) {
  Solution s;

  std::vector<int> nums1{1};
  std::vector<int> nums2{1};

  double median = s.findMedianSortedArrays(nums1, nums2);
  std::cout << median << std::endl;
}
