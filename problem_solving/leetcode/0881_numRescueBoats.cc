class Solution {
public:
  int numRescueBoats(vector<int>& people, int limit) {
    sort(people.begin(), people.end());

    int count = 0;
    int left = 0;
    int right = people.size() - 1;

    while (left <= right) {
      if (people[left] + people[right] > limit) {
        right--;
      } else {
        left++;
        right--;
      }
      count++;
    }

    return count;
  }
};
