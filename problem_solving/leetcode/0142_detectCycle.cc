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
  ListNode *detectCycle(ListNode *head) {
    ListNode *node = head;
    while (node != nullptr) {
      ListNode *nextNode = node->next;
      if (nextNode != nullptr && nextNode <= node) return nextNode;
      node = nextNode;
    }
    return NULL;
  }
};
