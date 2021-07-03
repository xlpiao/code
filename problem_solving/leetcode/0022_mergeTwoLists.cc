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
  ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode* n1 = l1;
    ListNode* n2 = l2;

    while(n2) {
      if(n1->val >=n2->val) {
        n1->next = n1;
        n1->val = n2->val;
        n2 = n2->next;
      }
      n1 = n1->next;
      cout << n1->val << ", ";
    }

    return n1;
  }
};
