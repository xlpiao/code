/**
 * File              : Graph.cc
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2021.05.22
 * Last Modified Date: 2021.05.22
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */

#include <iostream>
#include <vector>

using Adjacency = std::vector<std::vector<int>>;

class Graph {
public:
  Graph(int N) { adj_ = Adjacency(N); }

  //// Add an edge in an undirected graph.
  void addEdge(int i, int j) {
    adj_[i].push_back(j);
    adj_[j].push_back(i);
  }

  //// Print the adjacency list representation of graph
  void printGraph() {
    for (int n = 0; n < adj_.size(); ++n) {
      std::cout << "\n Adjacency list of vertex " << n << "\n head ";
      for (auto it : adj_[n]) {
        std::cout << " -> " << it;
      }
      std::cout << "\n";
    }
  }

private:
  Adjacency adj_;
};

int main(void) {
  int N = 5;
  Graph graph(N);

  graph.addEdge(0, 1);
  graph.addEdge(0, 4);
  graph.addEdge(1, 2);
  graph.addEdge(1, 3);
  graph.addEdge(1, 4);
  graph.addEdge(2, 3);
  graph.addEdge(3, 4);

  graph.printGraph();

  return 0;
}
