class Solution {
public:
  void moveZeroes(vector<int>& nums) {
    int curr = 0;

    for (int i = 0; i < nums.size(); i++) {
      if (nums[i] != 0) {
        if (i != curr) {
          swap(nums[i], nums[curr]);
        }
        curr++;
      }
    }
  }
};
