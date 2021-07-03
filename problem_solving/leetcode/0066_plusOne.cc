class Solution {
public:
  vector<int> plusOne(vector<int>& digits) {
    std::vector<int> ret;
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
      ret.push_back(carry);
    }
    for(int i = 0; i < digits.size(); i++) {
      ret.push_back(digits[i]);
    }
    return ret;
  }
};
