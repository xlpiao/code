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
  TreeNode* createBinaryTree(vector<vector<int>>& descriptions) {
    unordered_map<int, TreeNode*> parent_mp;
    unordered_map<int, int> child_mp;

    for (int i = 0; i < descriptions.size(); i++) {
      int parent = descriptions[i][0];
      int child = descriptions[i][1];
      int isLeft = descriptions[i][2];

      if (parent_mp.find(parent) == parent_mp.end()) {
        TreeNode* temp = new TreeNode(parent);
        parent_mp[parent] = temp;
      }
      if (parent_mp.find(child) == parent_mp.end()) {
        TreeNode* temp = new TreeNode(child);
        parent_mp[child] = temp;
      }
      if (isLeft == 1) {
        parent_mp[parent]->left = parent_mp[child];
      } else {
        parent_mp[parent]->right = parent_mp[child];
      }
      child_mp[child] = parent;
    }

    TreeNode* ans = NULL;
    for (int i = 0; i < descriptions.size(); i++) {
      int parent = descriptions[i][0];
      if (child_mp.find(parent) == child_mp.end()) {
        ans = parent_mp[parent];
      }
    }
    return ans;
  }
};
