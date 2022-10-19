/**
 * File              : Tree.cc
 * Author            : Xianglan Piao <lanxlpiao@gmail.com>
 * Date              : 2021.05.22
 * Last Modified Date: 2021.05.22
 * Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>
 */

#include <iostream>
#include <stack>
#include <vector>
using namespace std;

#define TEST 1

struct Node {
  int val;
  struct Node* left;
  struct Node* right;
};

Node* newNode(int val) {
  Node* temp = new Node;
  temp->val = val;
  temp->left = temp->right = NULL;
  cout << val << " ";
  return temp;
}

void preOrderDFS(struct Node* node) {
  if (node == NULL) return;

  cout << node->val << ",";
  preOrderDFS(node->left);
  preOrderDFS(node->right);
}

void inOrderDFS(struct Node* node) {
  if (node == NULL) return;

  inOrderDFS(node->left);
  cout << node->val << ",";
  inOrderDFS(node->right);
}

void postOrderDFS(struct Node* node) {
  if (node == NULL) return;

  postOrderDFS(node->left);
  postOrderDFS(node->right);
  cout << node->val << ",";
}

void preOrderStack(struct Node* node) {
  struct Node* n = node;
  stack<struct Node*> st;
  vector<int> ans;

#if TEST == 1
  st.push(n);
  while (!st.empty()) {
    while (n) {
      ans.push_back(n->val);
      st.push(n);
      n = n->left;
    }
    n = st.top();
    st.pop();

    n = n->right;
  }
#elif TEST == 2
  st.push(n);
  while (!st.empty()) {
    if (n) {
      ans.push_back(n->val);
      st.push(n);
      n = n->left;
    } else {
      n = st.top();
      st.pop();
      n = n->right;
    }
  }
#elif TEST == 3
  st.push(n);
  while (!st.empty()) {
    n = st.top();
    st.pop();
    ans.push_back(n->val);

    if (n->right) {
      st.push(n->right);
    }
    if (n->left) {
      st.push(n->left);
    }
  }
#endif

  for (auto it : ans) {
    cout << it << ",";
  }
}

void inOrderStack(struct Node* node) {
  struct Node* n = node;
  stack<struct Node*> st;
  vector<int> ans;

#if TEST == 1
  while (n || !st.empty()) {
    while (n) {
      st.push(n);
      n = n->left;
    }
    n = st.top();
    st.pop();
    ans.push_back(n->val);

    n = n->right;
  }
#elif TEST == 2
  while (n || !st.empty()) {
    if (n) {
      st.push(n);
      n = n->left;
    } else {
      n = st.top();
      ans.push_back(n->val);
      st.pop();
      n = n->right;
    }
  }
#elif TEST == 3
  cout << "no such solution\n";
#endif

  for (auto it : ans) {
    cout << it << ",";
  }
}

void postOrderStack(struct Node* node) {
  struct Node* n = node;
  stack<struct Node*> st;
  vector<int> ans;

#if TEST == 1
  while (n || !st.empty()) {
    while (n) {
      ans.push_back(n->val);
      st.push(n);
      n = n->right;
    }
    n = st.top();
    st.pop();

    n = n->left;
  }
#elif TEST == 2
  while (n || !st.empty()) {
    if (n) {
      ans.push_back(n->val);
      st.push(n);
      n = n->right;
    } else {
      n = st.top();
      st.pop();
      n = n->left;
    }
  }
#elif TEST == 3
  st.push(n);
  while (!st.empty()) {
    n = st.top();
    st.pop();
    ans.push_back(n->val);

    if (n->left) {
      st.push(n->left);
    }
    if (n->right) {
      st.push(n->right);
    }
  }
#endif

  reverse(ans.begin(), ans.end());
  for (auto it : ans) {
    cout << it << ",";
  }
}

int main() {
  cout << "new tree:\n ";
  cout << "   ";
  struct Node* root = newNode(7);
  cout << "\n";
  cout << "   / \\ \n";
  cout << "  ";
  root->left = newNode(5);
  cout << "  ";
  root->right = newNode(8);
  cout << "\n";
  cout << " / \\\n";
  root->left->left = newNode(2);
  cout << "  ";
  root->left->right = newNode(6);
  cout << "\n";

  cout << "\nApproach #" << TEST;
  cout << "\npreOrderDFS: ";
  preOrderDFS(root);
  cout << "\npreOrderStack: ";
  preOrderStack(root);

  cout << "\ninOrderDFS: ";
  inOrderDFS(root);
  cout << "\ninOrderStack: ";
  inOrderStack(root);

  cout << "\npostOrderDFS: ";
  postOrderDFS(root);
  cout << "\npostOrderStack: ";
  postOrderStack(root);
  cout << "\n" << endl;

  return 0;
}
