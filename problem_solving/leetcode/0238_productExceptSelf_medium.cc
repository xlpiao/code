class Solution {
public:
  vector<int> productExceptSelf(vector<int>& nums) {
    int n = nums.size();
    vector<int> answer(n, 1);

    int leftProduct = 1;
    int rightProduct = 1;

    for (int i = 0; i < n; i++) {
      answer[i] *= leftProduct;
      leftProduct *= nums[i];

      int j = n - i - 1;
      answer[j] *= rightProduct;
      rightProduct *= nums[j];
    }

    return answer;
  }
};

class Solution {
public:
  vector<int> productExceptSelf(vector<int>& nums) {
    int n = nums.size();
    vector<int> answer(n, 1);

    vector<int> leftProduct(n, 1);
    vector<int> rightProduct(n, 1);

    leftProduct[0] = 1;
    leftProduct[1] = nums[0];
    for (int i = 2; i < n; i++) {
      leftProduct[i] = nums[i - 1] * leftProduct[i - 1];
    }

    rightProduct[n - 1] = 1;
    rightProduct[n - 2] = nums[n - 1];
    for (int i = n - 3; i >= 0; i--) {
      rightProduct[i] = nums[i + 1] * rightProduct[i + 1];
      cout << rightProduct[i] << endl;
    }

    for (int i = 0; i < n; i++) {
      answer[i] = leftProduct[i] * rightProduct[i];
    }

    return answer;
  }
};

class Solution {
public:
  vector<int> productExceptSelf(vector<int>& nums) {
    int n = nums.size();

    int left = nums[0];
    int right = nums[n - 1];

    vector<int> res(n, 1);

    for (int i = 1; i < n; i++) {
      res[i] = left;
      left *= nums[i];
    }

    for (int i = n - 2; i >= 0; i--) {
      res[i] *= right;
      right *= nums[i];
    }

    return res;
  }
};
