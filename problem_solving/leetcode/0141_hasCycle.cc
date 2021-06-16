/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */

class Solution {
public:
  bool hasCycle(ListNode *head) {
    ListNode *node = head;
    while (node != nullptr) {
      ListNode *nextNode = node->next;
      // cout << node->val << endl;
      if (nextNode != nullptr && nextNode <= node) return true;
      node = nextNode;
    }
    return false;
  }
};

class Solution {
public:
  bool hasCycle(ListNode *head) {
    while (head) {
      m_[head]++;
      if (m_[head] >= 2) {
        return true;
      }
      head = head->next;
    }
    return false;
  }

private:
  unordered_map<ListNode *, int> m_;
};
