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
  ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    ListNode* t1 = l1;
    ListNode* t2 = l2;

    int value = (t1->val + t2->val) % 10;
    int carry = (t1->val + t2->val) / 10;
    ListNode* root = new ListNode(value);
    ListNode* temp = root;

    t1 = t1->next;
    t2 = t2->next;
    while (t1 != NULL || t2 != NULL) {
      if (t1 != NULL && t2 != NULL) {
        value = (t1->val + t2->val + carry) % 10;
        carry = (t1->val + t2->val + carry) / 10;
        t1 = t1->next;
        t2 = t2->next;
      } else if (t1 != NULL && t2 == NULL) {
        value = (t1->val + carry) % 10;
        carry = (t1->val + carry) / 10;
        t1 = t1->next;
      } else if (t1 == NULL && t2 != NULL) {
        value = (t2->val + carry) % 10;
        carry = (t2->val + carry) / 10;
        t2 = t2->next;
      }

      ListNode* next = new ListNode(value);
      temp->next = next;
      temp = temp->next;
    }

    if (carry == 1) {
      ListNode* next = new ListNode(carry);
      temp->next = next;
      temp = temp->next;
    }
    return root;
  }
};
