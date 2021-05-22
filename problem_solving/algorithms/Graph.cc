/**
 * File              : Graph.cc
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2021.05.22
 * Last Modified Date: 2021.05.22
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */

#include <iostream>
#include <queue>
#include <vector>

using List = std::vector<int>;
using Adjacency = std::vector<List>;

class Graph {
public:
  Graph(int N) {
    adj_ = Adjacency(N);
    visited_ = std::vector<bool>(N);
  }

  //// Add an edge in an undirected graph.
  void addEdge(int i, int j) {
    if (!exist(adj_[i], j)) {
      adj_[i].push_back(j);
    }
    if (!exist(adj_[j], i)) {
      adj_[j].push_back(i);
    }
  }

  bool exist(List l, int data) {
    for (auto it : l) {
      if (it == data) {
        return true;
      }
    }
    return false;
  }

  //// Print the adjacency list representation of graph
  void printGraph() {
    for (int n = 0; n < adj_.size(); ++n) {
      std::cout << "Adjacency List of Node[" << n << "]:";
      for (auto it : adj_[n]) {
        std::cout << " -> " << it;
      }
      std::cout << "\n";
    }
  }

  bool BFS(int start, int end) {
    for (auto it : visited_) {
      it = false;
    }

    std::cout << "\nPath " << start << " -> " << end << ": ";
    queue_.push(start);
    while (!queue_.empty()) {
      int head = queue_.front();
      queue_.pop();
      visited_[start] = true;
      std::cout << " -> " << head;
      if (head == end) {
        return true;
      }
      List l = adj_[head];
      for (int i = 0; i < l.size(); i++) {
        if (!visited_[l[i]]) {
          visited_[l[i]] = true;
          queue_.push(l[i]);
        }
      }
    }
    return false;
  }

private:
  Adjacency adj_;
  std::vector<bool> visited_;
  std::queue<int> queue_;
};

int main(void) {
  int N = 10;
  Graph graph(N);

  graph.addEdge(0, 1);
  graph.addEdge(0, 2);
  graph.addEdge(1, 6);
  graph.addEdge(2, 3);
  graph.addEdge(2, 1);
  graph.addEdge(3, 5);
  graph.addEdge(4, 7);
  graph.addEdge(3, 6);
  graph.addEdge(5, 7);
  graph.addEdge(5, 8);
  graph.addEdge(6, 8);

  graph.printGraph();

  bool found = graph.BFS(5, 8);
  std::cout << "\nFind Path: " << found << std::endl;

  return 0;
}
