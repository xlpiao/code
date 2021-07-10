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
  TreeNode* sortedArrayToBST(vector<int>& nums) {
    if (nums.size() == 0) return NULL;

    int n = nums.size() / 2;
    TreeNode* root = new TreeNode(nums[n]);
    vector<int> left{nums.begin(), nums.begin() + n};
    vector<int> right{nums.begin() + n + 1, nums.end()};
    root->left = sortedArrayToBST(left);
    root->right = sortedArrayToBST(right);

    return root;
  }
};
