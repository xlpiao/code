// C program for linked list implementation of stack
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

// A structure to represent a stack
struct StackNode {
  int data;
  struct StackNode* next;
};

struct Stack {
  int length;
  struct StackNode* root;
};

struct StackNode* newNode(int data) {
  struct StackNode* stackNode =
      (struct StackNode*)malloc(sizeof(struct StackNode));
  stackNode->data = data;
  stackNode->next = NULL;
  return stackNode;
}

int isEmpty(struct Stack* root) { return !root->; }

void push(struct StackNode** top, int data) {
  struct StackNode* stackNode = newNode(data);
  stackNode->next = *top;
  *top = stackNode;
  printf("%d pushed to stack\n", data);
}

int pop(struct StackNode** top) {
  if (isEmpty(*top)) return INT_MIN;
  struct StackNode* temp = *top;
  *top = (*top)->next;
  int popped = temp->data;
  free(temp);

  return popped;
}

int peek(struct StackNode* top) {
  if (isEmpty(top)) return INT_MIN;
  return top->data;
}

int main(void) {
  struct Stack* stack = (struct Stack*)malloc(sizeof(struct Stack));
  stack->length = 0;
  stack->root = NULL;

  push(stack->root, 10);
  push(stack->root, 20);
  push(stack->root, 30);

  printf("%d popped from stack->n", pop(stack->root));

  printf("Top element is %d\n", peek(stack->root));

  return 0;
}
