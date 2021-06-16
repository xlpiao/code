class Solution {
public:
  bool canJump(vector<int>& nums) {
    int len = nums.size();
    int reach = 0;

    for (int i = 0; i < len; i++) {
      if (reach >= len - 1) return true;
      if (reach < i) return false;
      reach = max(reach, i + nums[i]);
    }

    return true;
  }
};
