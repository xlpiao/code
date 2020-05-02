/**
 * File              : dfs.cpp
 * Author            : Xianglan Piao <xianglan0502@gmail.com>
 * Date              : 2018.08.11
 * Last Modified Date: 2018.08.11
 * Last Modified By  : Xianglan Piao <xianglan0502@gmail.com>
 */
// A simple representation of graph using STL
#include <iostream>
#include <vector>
using namespace std;

// A utility function to add an edge in an
// undirected graph.
void addEdge(vector<int> adj[], int u, int v) {
  adj[u].push_back(v);
  adj[v].push_back(u);
}

// A utility function to print the adjacency list
// representation of graph
void printGraph(vector<int> adj[], int V) {
  for (int v = 0; v < V; ++v) {
    std::cout << "\n Adjacency list of vertex " << v << "\n head ";
    // for (auto x : adj[v])
    for (std::vector<int>::iterator it = adj[v].begin(); it != adj[v].end();
         ++it)
      std::cout << " -> " << *it;
    std::cout << "\n";
  }
}

// Driver code
int main() {
  int V = 5;
  vector<int> adj[V];
  addEdge(adj, 0, 1);
  addEdge(adj, 0, 4);
  addEdge(adj, 1, 2);
  addEdge(adj, 1, 3);
  addEdge(adj, 1, 4);
  addEdge(adj, 2, 3);
  addEdge(adj, 3, 4);
  printGraph(adj, V);
  return 0;
}
