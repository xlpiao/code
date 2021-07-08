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
class Solution {
public:
  bool check(TreeNode* pLeft, TreeNode* pRight) {
    if (pLeft == nullptr || pRight == nullptr) return pLeft == pRight;

    if (pLeft->val != pRight->val) return false;

    return check(pLeft->left, pRight->right) &&
           check(pLeft->right, pRight->left);
  }

  bool isSymmetric(TreeNode* root) {
    if (root == nullptr) {
      return true;
    }

    return check(root->left, root->right);
  }
};
