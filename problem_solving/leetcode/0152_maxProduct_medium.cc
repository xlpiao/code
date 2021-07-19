class Solution {
public:
  int maxProduct(vector<int>& nums) {
    int max_so_far = nums[0];
    int min_so_far = nums[0];
    int result = nums[0];

    for (int i = 1; i < nums.size(); i++) {
      int curr_max = max({nums[i], max_so_far * nums[i], min_so_far * nums[i]});
      min_so_far = min({nums[i], max_so_far * nums[i], min_so_far * nums[i]});
      max_so_far = curr_max;
      result = max(result, max_so_far);
    }
    return result;
  }
};
