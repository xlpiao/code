class Solution {
public:
  vector<int> plusOne(vector<int>& digits) {
    int carry = 0;
    for (int i = digits.size()-1; i >= 0; i--) {
      if(i == digits.size() - 1) {
        digits[i] += 1;
        carry = digits[i] / 10;
        digits[i] = digits[i] % 10;
      }
      else{
        digits[i] += carry;
        carry = digits[i] / 10;
        digits[i] = digits[i] % 10;
      }
    }
    
    if (carry != 0) {
      digits.insert(digits.begin(), carry);
    }
    
    return digits;
  }
};
