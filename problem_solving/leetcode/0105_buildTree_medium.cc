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
  TreeNode* build(vector<int>& preorder,
                  vector<int>& inorder,
                  int& index,
                  int l,
                  int r) {
    if (l > r) {
      return NULL;
    }

    TreeNode* root = new TreeNode(preorder[index]);

    int pivot = 0;
    for (int i = 0; i < inorder.size(); i++) {
      if (preorder[index] == inorder[i]) {
        pivot = i;
        break;
      }
    }

    index++;

    cout << index << ", " << l << "->" << pivot - 1 << endl;
    cout << index << ", " << pivot + 1 << "->" << r << endl;

    root->left = build(preorder, inorder, index, l, pivot - 1);
    root->right = build(preorder, inorder, index, pivot + 1, r);

    return root;
  }

  TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    int index = 0;
    return build(preorder, inorder, index, 0, inorder.size() - 1);
  }
};
