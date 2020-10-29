# Graph passing

Graph is an input data structure for graph neural network. Passing graph into GNN layer can have different behaviors. If graph remains fixed across samples, that is, all samples utilize the same graph structure, a static graph is used. Graphs can be carried within `FeaturedGraph` to provide variable graph to GNN layer. Users have the flexibility to pick a adequate approach for their own needs.

## Static graph

A static graph is used to reduce redundant computation during passing through layers. A static graph can be set in graph convolutional layers in prior such that graph in layers is used first for computation. A adjacency matrix `adj_mat` is given to represent a graph and is put into a graph convolutional layer as follow:

```
GCNConv(adj_mat, feat=>h1, relu)
```

`Simple(Di)Graph`, `SimpleWeighted(Di)Graph` or `Meta(Di)Graph` provided by LightGraphs, SimpleWeightedGraphs and MetaGraphs, respectively, are acceptable for passing to layer as a static graph. A adjacency list is also accepted in the type of `Vector{Vector}`.

## Variable graph

A variable graph is supported by `FeaturedGraph`. Each `FeaturedGraph` contains different graph structure and its features. Data of `FeaturedGraph` are feed directly to graph convolutional layer or graph neural network to let each feature be learn on different graph structure. A adjacency matrix `adj_mat` is given to construct a `FeaturedGraph` as follow:

```
FeaturedGraph(adj_mat, features)
```

`Simple(Di)Graph`, `SimpleWeighted(Di)Graph` or `Meta(Di)Graph` provided by LightGraphs, SimpleWeightedGraphs and MetaGraphs, respectively, are acceptable for constructing a `FeaturedGraph`. A adjacency list is also accepted, too.

## Cached graph in layers

While a variable graph is given by `FeaturedGraph`, a GNN layer don't need a static graph anymore. Besides taking off the static graph from arguments of a layer, remember to turn off the cache mechanism. A cache mechanism is designed to cache static graph to reduce computation. A cached graph is gotten from layer and computation is then performed. For each time, it will assign current computed graph back to layer. Assignment operation is not differentiable, so we must turn off the cache mechanism as follow:

```
GCNConv(feat=>h1, relu, cached=false)
```

This ensures layer function as expected.
