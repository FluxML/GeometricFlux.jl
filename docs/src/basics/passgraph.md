# Graph Passing Strategy

Graph is an input data structure for graph neural network. Passing a graph into a GNN layer can have different behaviors. If the graph remains fixed across samples, that is, all samples utilize the same graph structure, a static graph is used. Or, graphs can be carried within `FeaturedGraph` to provide variable graphs to GNN layer. Users have the flexibility to pick an adequate approach for their own needs.

## Variable Graph Strategy

Variable graphs are supported through `FeaturedGraph`, which contains both the graph information and the features. Each `FeaturedGraph` can contain a distinct graph structure and its features. Data of `FeaturedGraph` are fed directly to graph convolutional layer or graph neural network to let each feature be learned on different graph structures. A adjacency matrix `adj_mat` is given to construct a `FeaturedGraph` as follows:

```
fg = FeaturedGraph(adj_mat, features)
layer = GCNConv(feat=>h1, relu)
```

`Simple(Di)Graph`, `SimpleWeighted(Di)Graph` or `Meta(Di)Graph` provided by the packages Graphs, SimpleWeightedGraphs and MetaGraphs, respectively, are acceptable for constructing a `FeaturedGraph`. An adjacency list is also accepted, too.

### `FeaturedGraph` in, `FeaturedGraph` out

Since a variable graph is provided from data, a `FeaturedGraph` object or a set of `FeaturedGraph` objects should be fed in a GNN model. The `FeaturedGraph` object should contain a graph and sufficient features that a GNN model needed. After operations, a `FeaturedGraph` object is given as output.

```julia
fg = FeaturedGraph(g, nf=X)
gc = GCNConv(in_channel=>out_channel)
new_fg = gc(fg)
```

## Static Graph Strategy

A static graph is used to reduce redundant computation during passing through layers. A static graph can be set in graph convolutional layers such that this graph is shared for computations across those layers. An adjacency matrix `adj_mat` is given to represent a graph and is provided to a graph convolutional layer as follows:

```
fg = FeaturedGraph(adj_mat)
layer = WithGraph(fg, GCNConv(feat=>h1, relu))
```

`Simple(Di)Graph`, `SimpleWeighted(Di)Graph` or `Meta(Di)Graph` provided by the packages Graphs, SimpleWeightedGraphs and MetaGraphs, respectively, are valid arguments for passing as a static graph to this layer. An adjacency list is also accepted in the type of `Vector{Vector}` is also accepted.

### Cached Graph in Layers

While a variable graph is given by `FeaturedGraph`, a GNN layer doesn't need a static graph anymore. A cache mechanism is designed to cache static graph to reduce computation time. A cached graph is retrieved from `WithGraph` layer and operation is then performed. For each time, it will assign current computed graph back to layer.

### Array in, Array out

Since a static graph is provided from `WithGraph` layer, it doesn't accept a `FeaturedGraph` object anymore. Instead, it accepts a regular array as input, and outputs an array back.

```julia
fg = FeaturedGraph(g)
layer = WithGraph(fg, GCNConv(in_channel=>out_channel))
H = layer(X)
```

## What you feed is what you get

In GeometricFlux, there are are two APIs which allow different input/output types for GNN layers. For example, `GCNConv` layer provides the following two APIs:

```julia
(g::WithGraph{<:GCNConv})(X::AbstractArray) -> AbstractArray
(g::GCNConv)(fg::FeaturedGraph) -> FeaturedGraph
```

If your feed a `GCNConv` layer with a `Array`, it will return you a `Array`. If you feed a `GCNConv` layer with a `FeaturedGraph`, it will return you a `FeaturedGraph`. **These APIs ensure the consistency between input and output types.**
