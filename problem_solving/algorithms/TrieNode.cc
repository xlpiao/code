#include <iostream>

using namespace std;

class TrieNode {
public:
  TrieNode() {
    for (int i = 0; i < 26; i++) {
      child[i] = NULL;
    }
    data = "";
    isEnd = false;
  }

  void insert(string word) {
    TrieNode *node = this;

    for (char c : word) {
      int idx = c - 'a';
      if (node->child[idx] == NULL) {
        node->child[idx] = new TrieNode();
      }
      node = node->child[idx];
    }
    node->data = word;
    node->isEnd = true;
    cout << "insert: " << word << ", " << node->isEnd << "\n";
  }

  bool search(string word) {
    TrieNode *node = this;

    for (char c : word) {
      int idx = c - 'a';
      if (node->child[idx] == NULL) {
        return false;
      }
      node = node->child[idx];
    }
    cout << "search: " << word << ", " << node->isEnd << "\n";
    return node->isEnd;
  }

  bool startWith(string prefix) {
    TrieNode *node = this;

    for (char c : prefix) {
      int idx = c - 'a';
      if (node->child[idx] == NULL) {
        return false;
      }
      node = node->child[idx];
    }
    cout << "startWith: " << prefix << ", 1"
         << "\n";
    return true;
  }

private:
  TrieNode *child[26];
  string data;
  bool isEnd;
};

int main(void) {
  TrieNode node;

  node.insert("apple");
  node.search("apple");
  node.search("app");
  node.startWith("app");
  node.insert("app");
  node.search("app");

  return 0;
}
