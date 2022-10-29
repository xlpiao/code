#include <iostream>
#include <unordered_map>

using namespace std;

class TrieNode {
public:
  TrieNode() {
    for (char c = 'a'; c < 'z'; c++) {
      child[c] = NULL;
    }
    data = "";
    isEnd = false;
  }

  void insert(string word) {
    TrieNode *node = this;

    for (char c : word) {
      if (node->child[c] == NULL) {
        node->child[c] = new TrieNode();
      }
      node = node->child[c];
    }
    node->data = word;
    node->isEnd = true;
    cout << "insert: " << word << ", " << node->isEnd << "\n";
  }

  bool search(string word) {
    TrieNode *node = this;

    for (char c : word) {
      if (node->child[c] == NULL) {
        return false;
      }
      node = node->child[c];
    }
    cout << "search: " << word << ", " << node->isEnd << "\n";
    return node->isEnd;
  }

  bool startWith(string prefix) {
    TrieNode *node = this;

    for (char c : prefix) {
      if (node->child[c] == NULL) {
        return false;
      }
      node = node->child[c];
    }
    cout << "startWith: " << prefix << ", 1"
         << "\n";
    return true;
  }

private:
  unordered_map<char, TrieNode *> child;
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
