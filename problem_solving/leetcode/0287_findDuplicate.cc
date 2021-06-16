class Solution {
public:
  int findDuplicate(vector<int>& nums) {
    int duplicate;
    for (int i = 0; i < nums.size(); i++) {
      m_[nums[i]]++;
      if (m_[nums[i]] >= 2) {
        duplicate = nums[i];
        break;
      }
    }
    return duplicate;
  }

private:
  unordered_map<int, int> m_;
};
