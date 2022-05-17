# Dynamic Graph Update

Dynamic graph update is a technique to generate a new graph within a graph convolutional layer proposed by [Wang2019](@cite).

Most of manifold learning approaches aims to learn capture manifold structures in high dimensional space. They construct a graph to approximate the manifold and learn to reduce dimensions of space. The separation of capturing manifold and learning dimensional reduction limits the power of manifold learning. Thus, [latent graph learning](https://towardsdatascience.com/manifold-learning-2-99a25eeb677d) is proposed to learn the manifold and dimensional reduction simultaneously. The latent graph learning is also named as manifold learning 2.0 which leverages the power of graph neural network and learns latent graph structure within layers of a graph neural network.

Latent graph learning learns the latent graph through training over point cloud, or a set of features. A fixed graph structure is not provided to a GNN model. Latent graph is dynamically constructed by constructing a neighborhood graph using features in graph convolutional layers. After construction of neighborhood graph, the neighborhood graph is fed as input with features into a graph convolutional layer.

Currently, we support k-nearest neighbor method to construct a neighborhood graph. To use dynamic graph update, just replace the static graph strategy

```julia
WithGraph(fg, EdgeConv(Dense(2*in_channel, out_channel)))
```

as graph construction method.

```julia
WithGraph(
    EdgeConv(Dense(2*in_channel, out_channel)),
    dynamic=X -> GraphSignals.kneighbors_graph(X, 3)
)
```
