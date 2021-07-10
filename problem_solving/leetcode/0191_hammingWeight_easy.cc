class Solution {
public:
  int hammingWeight(uint32_t n) {
    int count = 0;

    while (n) {
      //// bin format: n & 0b01 is also ok
      //// hex format: n & 0x01 is also ok
      if (n & 1) {
        count++;
      }
      n = n >> 1;
    }

    return count;
  }
};
