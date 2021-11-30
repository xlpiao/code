class Solution {
public:
  vector<int> topKFrequent(vector<int>& nums, int k) {
    map<int, int> mp;
    vector<int> res;
    for (auto i : nums) {
      mp[i]++;
    }
    multimap<int, int, greater<int>> mmp;
    for (auto i : mp) {
      mmp.insert({i.second, i.first});
    }
    int c = 0;
    for (auto i : mmp) {
      res.push_back(i.second);
      c++;
      if (c == k) {
        break;
      }
    }
    return res;
  }
};
