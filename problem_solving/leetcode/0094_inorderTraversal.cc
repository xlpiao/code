/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left),
 * right(right) {}
 * };
 */

#include <iostream>
#include <string>
#include <vector>

using namespace std;

struct TreeNode {
  int val;
  struct TreeNode* left;
  struct TreeNode* right;

  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode* left, TreeNode* right)
      : val(x), left(left), right(right) {}
  void print(TreeNode* root) {
    if (root == nullptr) return;
    cout << root->val << endl;
    if (root->left) {
      cout << root->val << ":L";
      print(root->left);
    }
    if (root->right) {
      cout << root->val << ":R";
      print(root->right);
    }
  }
};

class Solution {
public:
  vector<int> inorderTraversal(TreeNode* root) {
    if (root == NULL) {
      return result;
    }

    inorderTraversal(root->left);
    result.push_back(root->val);
    inorderTraversal(root->right);

    return result;
  }

private:
  vector<int> result;
};

int main(void) {
  Solution s;

  struct TreeNode* root = new TreeNode(1);
  root->right = new TreeNode(2);
  root->right->left = new TreeNode(3);
  root->print(root);

  auto ret = s.inorderTraversal(root);
  for (auto it : ret) {
    cout << it << ", ";
  }
  cout << endl;

  return 0;
}
