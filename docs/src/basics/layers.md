# Building graph neural networks

Building GNN is as simple as building neural network in Flux. The syntax here is the same as Flux. `Chain` is used to stack layers into a GNN. A simple example is shown here:

```
model = Chain(GCNConv(adj_mat, feat=>h1),
              GCNConv(adj_mat, h1=>h2, relu))
```

In the example above, The first argument `adj_mat` is the representation of a graph in form of adjacency matrix. The feature dimension in first layer is mapped from `feat` to `h1`. In second layer, `h1` is then mapped to `h2`. Default activation function is given as identity if it is not specified by users.

The initialization function `GCNConv(...)` constructs a `GCNConv` layer. For most of the layer types in GeometricFlux, a layer can be initialized in at least two ways:

* Initializing *with* a predefined adjacency matrix or `FeaturedGraph`, followed by the other parameters. For most of the layer types, this is for datasets where each input has the same graph structure.
* Initializing *without* an initial graph argument, only supplying the relevant parameters. This allows the layer to accept different graph structures.

# Applying layers

When using GNN layers, the general guidelines are:

* If you pass in a ``n \times d`` matrix of node features, and the layer maps node features ``\mathbb R^d \rightarrow \mathbb R^k`` then the output will be in matrix with dimensions ``n \times k``. The same ostensibly goes for edge features but as of now no layer type supports outputting new edge features.
* If you pass in a `FeaturedGraph`, the output will be also be a `FeaturedGraph` with modified node (and/or edge) features. Add `node_feature` as the following entry in the Flux chain (or simply call `node_feature()` on the output) if you wish to subsequently convert them to matrix form.


## Customize layers

Customizing your own GNN layers are the same as customizing layers in Flux. You may want to reference [Flux documentation](https://fluxml.ai/Flux.jl/stable/models/basics/#Building-Layers-1).
