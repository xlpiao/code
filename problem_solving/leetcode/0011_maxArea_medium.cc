class Solution {
public:
  int maxArea(vector<int>& height) {
    int left = 0;
    int right = height.size() - 1;
    int mArea = INT_MIN;

    while (left < right) {
      int minValue = min(height[left], height[right]);
      mArea = max(mArea, (right - left) * minValue);

      if (height[left] <= height[right]) {
        left++;
      } else {
        right--;
      }
    }

    return mArea;
  }
};
