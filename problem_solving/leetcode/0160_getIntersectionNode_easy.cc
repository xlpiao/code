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
  ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    ListNode *a = headA;
    ListNode *b = headB;

    while (a != b) {
      if (a == nullptr) {
        a = headB;
      }
      if (b == nullptr) {
        b = headA;
      }
      if (a == b) {
        break;
      } else {
        a = a->next;
        b = b->next;
      }
    }

    return a;
  }
};
