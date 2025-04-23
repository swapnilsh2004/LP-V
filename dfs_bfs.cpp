#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
#include <chrono>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace std::chrono;

class Graph {
public:
    int V;
    vector<vector<int>> adj;

    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int v, int w) {
        adj[v].push_back(w);
        adj[w].push_back(v); 
    }

    void generateRandomEdges(int edges) {
        srand(time(0));
        for (int i = 1; i < V; i++) {
            int u = i;
            int v = rand() % i;
            addEdge(u, v);
        }
        for (int i = V - 1; i < edges; i++) {
            int u = rand() % V;
            int v = rand() % V;
            if (u != v) {
                addEdge(u, v);
            }
        }
    }

    void BFSSequential(int start) {
        vector<bool> visited(V, false);
        queue<int> q;
        q.push(start);
        visited[start] = true;

        cout << "BFS Sequential: ";
        while (!q.empty()) {
            int node = q.front();
            q.pop();
            cout << node << " ";

            for (int neighbor : adj[node]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
        cout << endl;
    }

    void BFSParallel(int start) {
        vector<bool> visited(V, false);
        vector<int> frontier;
        frontier.push_back(start);
        visited[start] = true;

        cout << "BFS Parallel: ";
        while (!frontier.empty()) {
            vector<int> nextFrontier;
            #pragma omp parallel for
            for (size_t i = 0; i < frontier.size(); i++) {
                int node = frontier[i];
                cout << node << " ";  
                for (int neighbor : adj[node]) {
                    if (!visited[neighbor]) {
                        #pragma omp critical
                        {
                            if (!visited[neighbor]) {
                                visited[neighbor] = true;
                                nextFrontier.push_back(neighbor);
                            }
                        }
                    }
                }
            }
            frontier = move(nextFrontier);
        }
        cout << endl;
    }

    void DFSSequential(int start) {
        vector<bool> visited(V, false);
        stack<int> s;
        s.push(start);

        cout << "DFS Sequential: ";
        while (!s.empty()) {
            int node = s.top();
            s.pop();
            if (!visited[node]) {
                visited[node] = true;
                cout << node << " ";
                for (int neighbor : adj[node]) {
                    if (!visited[neighbor]) {
                        s.push(neighbor);
                    }
                }
            }
        }
        cout << endl;
    }

    void DFSParallelUtil(int node, vector<bool> &visited) {
        visited[node] = true;
        cout << node << " ";

        #pragma omp parallel for
        for (size_t i = 0; i < adj[node].size(); i++) {
            int neighbor = adj[node][i];
            if (!visited[neighbor]) {
                #pragma omp task
                DFSParallelUtil(neighbor, visited);
            }
        }
    }

    void DFSParallel(int start) {
        vector<bool> visited(V, false);
        cout << "DFS Parallel: ";
        #pragma omp parallel
        {
            #pragma omp single
            DFSParallelUtil(start, visited);
        }
        cout << endl;
    }
};

int main() {
    int nodes, edges;
    cout << "Enter number of nodes: ";
    cin >> nodes;
    cout << "Enter number of edges: ";
    cin >> edges;

    Graph g(nodes);
    g.generateRandomEdges(edges);

    auto start = high_resolution_clock::now();
    g.BFSSequential(0);
    auto stop = high_resolution_clock::now();
    auto durationBFSSeq = duration_cast<microseconds>(stop - start);

    start = high_resolution_clock::now();
    g.BFSParallel(0);
    stop = high_resolution_clock::now();
    auto durationBFSPar = duration_cast<microseconds>(stop - start);

    start = high_resolution_clock::now();
    g.DFSSequential(0);
    stop = high_resolution_clock::now();
    auto durationDFSSeq = duration_cast<microseconds>(stop - start);

    start = high_resolution_clock::now();
    g.DFSParallel(0);
    stop = high_resolution_clock::now();
    auto durationDFSPar = duration_cast<microseconds>(stop - start);

    cout << "BFS Sequential: " << durationBFSSeq.count()  << " microseconds" << endl;
    cout << "BFS Parallel: " << durationBFSPar.count()  << " microseconds" << endl;
    cout << "DFS Sequential: " << durationDFSSeq.count()  << " microseconds" << endl;
    cout << "DFS Parallel: " << durationDFSPar.count()  << " microseconds" << endl;
    
    cout << "BFS Speedup Factor: " << (double)durationBFSSeq.count() / durationBFSPar.count() << endl;
    cout << "DFS Speedup Factor: " << (double)durationDFSSeq.count() / durationDFSPar.count() << endl;

    return 0;
}