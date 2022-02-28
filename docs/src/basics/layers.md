# Building Graph Neural Networks

Building GNN is as simple as building neural network in Flux. The syntax here is the same as Flux. `Chain` is used to stack layers into a GNN. A simple example is shown here:

```julia
model = Chain(
    GCNConv(feat=>h1),
    GCNConv(h1=>h2, relu),
)
```

In the example above, the feature dimension in first layer is mapped from `feat` to `h1`. In second layer, `h1` is then mapped to `h2`. Default activation function is given as `identity` if it is not specified by users.

The initialization function `GCNConv(...)` constructs a `GCNConv` layer. For most of the layer types in GeometricFlux, a layer can be initialized in two ways:

* GNN layer without graph: initializing *without* a predefined graph topology. This allows the layer to accept different graph topology.
* GNN layer with static graph: initializing *with* a predefined graph topology, e.g. graph wrapped in `FeaturedGraph`. This strategy is suitable for datasets where each input requires the same graph structure and it has better performance than variable graph strategy.

The example above demonstrate the variable graph strategy. The equivalent GNN architecture but with static graph strategy is shown as following:

```julia
model = Chain(
    WithGraph(fg, GCNConv(feat=>h1)),
    WithGraph(fg, GCNConv(h1=>h2, relu)),
)
```

```@docs
GeometricFlux.WithGraph
```

## Applying Layers

When using GNN layers, the general guidelines are:

* With static graph strategy: you should pass in a ``d \times n \times batch`` matrix for node features, and the layer maps node features ``\mathbb{R}^d \rightarrow \mathbb{R}^k`` then the output will be in matrix with dimensions ``k \times n \times batch``. The same ostensibly goes for edge features but as of now no layer type supports outputting new edge features.
* With variable graph strategy: you should pass in a `FeaturedGraph`, the output will be also be a `FeaturedGraph` with modified node (and/or edge) features. Add `node_feature` as the following entry in the Flux chain (or simply call `node_feature()` on the output) if you wish to subsequently convert them to matrix form.

## Create Custom GNN Layers

Customizing your own GNN layers are the same as customizing layers in Flux. You may want to reference [Flux documentation](https://fluxml.ai/Flux.jl/stable/models/basics/#Building-Layers-1).
