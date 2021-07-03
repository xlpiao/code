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
    if(!l1 && !l2) return NULL;
    
    ListNode* n = new ListNode();
    ListNode *root = n;
      
    
    while(l1 || l2) {
      if(l1 && l2) {
        if(l1->val <= l2->val) {
          n->val = l1->val;
          n->next = new ListNode();
          l1 = l1->next;
        }
        else{
          n->val = l2->val;
          n->next = new ListNode();
          l2 = l2->next;
        }
      }
      else if (l1) {
        n->val = l1->val;
        n->next = l1->next;
        break;
      }
      else if(l2) {
        n->val = l2->val;
        n->next = l2->next;
        break;
      }
      cout << n->val << ", ";
      n = n->next;
    }
    return root;
  }
};
