class Solution {
public:
  int firstMissingPositive(vector<int>& nums) {
    if (nums.size() == 1) {
      if (nums[0] == 1)
        return 2;
      else
        return 1;
    }

    for (int i = 0; i < nums.size(); i++) {
      if (nums[i] >= 1) {
        s_.insert(nums[i]);
      }
    }
    for (int i = 1; i <= s_.size(); i++) {
      if (s_.count(i) == 0) return i;
    }

    return s_.size() + 1;
  }

private:
  set<int> s_;
};
