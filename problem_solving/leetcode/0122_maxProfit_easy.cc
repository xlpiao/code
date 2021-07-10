class Solution {
public:
  int maxProfit(vector<int>& prices) {
    int n = prices.size();

    int profit = 0;
    for (int i = 1; i < n; i++) {
      profit += max(prices[i] - prices[i - 1], 0);
    }

    return profit;
  }
};
