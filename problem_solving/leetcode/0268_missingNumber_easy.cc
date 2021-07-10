class Solution {
public:
  int missingNumber(vector<int>& nums) {
    int n = nums.size();
    int correctSum = (n * (n + 1)) / 2;

    int sum = 0;
    for (int i = 0; i < n; i++) {
      sum += nums[i];
    }

    return (correctSum - sum);
  }
};

class Solution {
public:
  int missingNumber(vector<int>& nums) {
    int n = nums.size();
    //// arithmetic sequence: a{n}=a{1}+(n-1)d;
    int correctSum = (n * (n + 1)) / 2;

    for (int i = 0; i < n; i++) {
      correctSum -= nums[i];
    }

    return correctSum;
  }
};
