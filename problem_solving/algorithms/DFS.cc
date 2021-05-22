/**
 * File              : dfs.cpp
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2018.08.11
 * Last Modified Date: 2018.08.11
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */

#include <iostream>
#include <vector>

using Adjacency = std::vector<std::vector<int>>;

// Add an edge in an undirected graph.
void addEdge(Adjacency &adj, int i, int j) {
  adj[i].push_back(j);
  adj[j].push_back(i);
}

// Print the adjacency list representation of graph
void printGraph(Adjacency &adj, int N) {
  for (int n = 0; n < N; ++n) {
    std::cout << "\n Adjacency list of vertex " << n << "\n head ";
    for (auto it : adj[n]) {
      std::cout << " -> " << it;
    }
    std::cout << "\n";
  }
}

int main(void) {
  int N = 5;
  Adjacency adj(N);

  addEdge(adj, 0, 1);
  addEdge(adj, 0, 4);
  addEdge(adj, 1, 2);
  addEdge(adj, 1, 3);
  addEdge(adj, 1, 4);
  addEdge(adj, 2, 3);
  addEdge(adj, 3, 4);

  printGraph(adj, N);

  return 0;
}
