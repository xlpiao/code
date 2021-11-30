class Solution {
public:
  int gcd(int a, int b) {
    if (b == 0) return a;
    return gcd(b, a % b);
  }

  string gcdOfStrings(string s1, string s2) {
    if (s1 + s2 == s2 + s1) {
      int t = gcd(s1.length(), s2.length());
      return s1.substr(0, t);
    }
    return "";
  }
};
