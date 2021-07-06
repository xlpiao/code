#include <iostream>
#include <vector>

using namespace std;

#if 0
// Time Limit Exceeded
class Solution {
public:
  int countPrimes(int n) {
    std::vector<int> primes;

    if (n <= 2) return 0;

    if (n > 2) primes.push_back(2);

    for (int i = 3; i < n; i++) {
      int j = 0;
      for (; j < primes.size(); j++) {
        if (i != primes[j] && i % primes[j] == 0) {
          break;
        }
      }
      if (j == primes.size()) {
        primes.push_back(i);
      }
    }

    return primes.size();
  }
};
#endif

#if 0
// Solution #1
class Solution {
public:
  int countPrimes(int n) {
    std::vector<bool> primes(n, true);

    if (n <= 2) return 0;

    for (int i = 2; i * i <= n; i++) {
      if (primes[i] == true) {
        for (int j = i * i; j <= n; j += i) primes[j] = false;
      }
    }

    int count = 0;
    for (int k = 2; k < n; k++) {
      if (primes[k]) count++;
    }

    return count;
  }
};
#endif

/* COMMENT: Sieve of Eratosthenes:
 * https://www.geeksforgeeks.org/sieve-of-eratosthenes/
 */
class Solution {
public:
  int countPrimes(int n) {
    std::vector<bool> primes(n, true);
    int count = n - 2;

    if (n <= 2) return 0;

    for (int i = 2; i * i < n; i++) {
      if (primes[i] == true) {
        for (int j = i * i; j < n; j += i) {
          if (primes[j]) {
            count--;
            primes[j] = false;
          }
        }
      }
    }

    return count;
  }
};

int main(void) {
  Solution s;
  cout << s.countPrimes(5) << endl;

  return 0;
}
