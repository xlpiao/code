/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
  bool isPalindrome(ListNode* head) {
    unsigned long long reversed = 0;
    unsigned long long number = 0;
    unsigned long long y = 1;

    ListNode* node = head;
    while (node != NULL) {
      number = number * 10 + node->val;
      reversed = reversed + node->val * y;
      y = y * 10;
      node = node->next;
    }
    return reversed == number;
  }
};

class Solution {
public:
  bool isPalindrome(ListNode* head) {
    vector<int> num;

    while (head != NULL) {
      num.push_back(head->val);
      head = head->next;
    }

    for (int i = 0; i < num.size() / 2; i++) {
      if (num[i] != num[num.size() - i - 1]) {
        return false;
      }
    }
    return true;
  }
};
