#include <iostream>

#define LOG() printf("%s +%d\t%s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__)

using namespace std;

class ListNode {
 public:
  ListNode(int data) : data{data} {
    LOG();
    next = nullptr;
  }
  ListNode(int data, ListNode* next) : data{data}, next{next} { LOG(); }

  int data;
  ListNode* next;
};

int main(void) {
  ListNode* one = new ListNode(1);
  ListNode* two = new ListNode(2);
  ListNode* three = new ListNode(3, nullptr);

  one->next = two;
  two->next = three;

  // print the linked list data
  ListNode* head = one;
  while (head != nullptr) {
    cout << head->data << " -> ";
    head = head->next;
  }

  return 0;
}
