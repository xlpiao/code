#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  int maxProfit(vector<int>& prices) {
    int buy = prices[0];
    int profit = 0;

    for (int i = 1; i < prices.size(); i++) {
      buy = min(buy, prices[i]);
      profit = max(profit, prices[i] - buy);
    }
    return profit;
  }
};

int main(void) {
  Solution s;

  vector<int> prices{7, 1, 5, 3, 6, 4};
  cout << s.maxProfit(prices) << endl;
}