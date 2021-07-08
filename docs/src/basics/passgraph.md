# Graph passing

Graph is an input data structure for graph neural network. Passing a graph into GNN layer can have different behaviors. If the graph remains fixed across samples, that is, all samples utilize the same graph structure, a static graph is used. Graphs can be carried within `FeaturedGraph` to provide variable graphs to GNN layer. Users have the flexibility to pick an adequate approach for their own needs.

## Static graph

A static graph is used to reduce redundant computation during passing through layers. A static graph can be set in graph convolutional layers such that this graph is shared for computations across those layers. An adjacency matrix `adj_mat` is given to represent a graph and is provided to a graph convolutional layer as follows:

```
GCNConv(adj_mat, feat=>h1, relu)
```

`Simple(Di)Graph`, `SimpleWeighted(Di)Graph` or `Meta(Di)Graph` provided by the packages LightGraphs, SimpleWeightedGraphs and MetaGraphs, respectively, are valid arguments for passing as a static graph to this layer. An adjacency list in the type of `Vector{Vector}` is also accepted .

## Variable graph

Variable graphs are supported through `FeaturedGraph`, which contains both the graph information and the features. Each `FeaturedGraph` can contain a different graph structure and its features. Data of `FeaturedGraph` are directly fed to graph convolutional layer or graph neural network to let each feature be learned on different graph structures. An adjacency matrix `adj_mat` is given to construct a `FeaturedGraph` as follow:

```
FeaturedGraph(adj_mat, features)
```

`Simple(Di)Graph`, `SimpleWeighted(Di)Graph` or `Meta(Di)Graph` provided by the packages LightGraphs, SimpleWeightedGraphs and MetaGraphs, respectively, are acceptable for constructing a `FeaturedGraph`. An adjacency list is also accepted, too.

## Cached graph in layers

While a variable graph is given by `FeaturedGraph`, a GNN layer don't need a static graph anymore. When no graph or adjacency matrix is passed as argument to a layer, caching will be disabled. The static graph cache mechanism is designed to reduce computation. A cached graph is revieved from the GNN layer and computation is then performed. For each time, it will assign current computed graph back to layer. This assignment operation is not differentiable, so the cache mechanism is turned off.