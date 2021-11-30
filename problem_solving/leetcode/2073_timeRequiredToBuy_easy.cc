class Solution {
public:
  int timeRequiredToBuy(vector<int>& tickets, int k) {
    int len = tickets.size();
    int time = 0;
    int i = 0;
    while (tickets[k] > 0) {
      --tickets[i];
      if (tickets[i] >= 0) ++time;
      i++;
      if (i % len == 0) i = 0;
    }
    return time;
  }
};
