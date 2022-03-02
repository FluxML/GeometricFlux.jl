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

## Define Your Own GNN Layer

Customizing your own GNN layers are the same as defining a layer in Flux. You may want to check [Flux documentation](https://fluxml.ai/Flux.jl/stable/models/basics/#Building-Layers-1) first.

To define a customized GNN layer, for example, we take a simple `GCNConv` layer as example here.

```julia
struct GCNConv <: AbstractGraphLayer
    weight
    bias
    σ
end

@functor GCNConv
```

We first should define a `GCNConv` type and let it be the subtype of `AbstractGraphLayer`. In this type, it holds parameters that a layer operate on. Don't forget to add `@functor` macro to `GCNConv` type.

```julia
(l::GCNConv)(Ã::AbstractMatrix, x::AbstractMatrix) = l.σ.(l.weight * x * Ã .+ l.bias)
```

Then, we can define the operation for `GCNConv` layer.

```julia
function (l::GCNConv)(fg::AbstractFeaturedGraph)
    nf = node_feature(fg)
    Ã = Zygote.ignore() do
        GraphSignals.normalized_adjacency_matrix(fg, eltype(nf); selfloop=true)
    end
    return ConcreteFeaturedGraph(fg, nf = l(Ã, nf))
end
```

Here comes to the GNN-specific behaviors. A GNN layer should accept object of subtype of `AbstractFeaturedGraph` to support variable graph strategy. A variable graph strategy should fetch node/edge/global features from `fg` and transform graph in `fg` into required form for layer operation, e.g. `GCNConv` layer needs a normalized adjacency matrix with self loop. Then, normalized adjacency matrix `Ã` and node features `nf` are pass through `GCNConv` layer `l(Ã, nf)` to give a new node feature. Finally, a `ConcreteFeaturedGraph` wrap graph in `fg` and new node features into a new object of subtype of `AbstractFeaturedGraph`.

```julia
layer = GCNConv(10=>5, relu)
new_fg = layer(fg)
gradient(() -> sum(node_feature(layer(fg))), Flux.params(layer))
```

Now we complete a simple version of `GCNConv` layer. One can test the forward pass and gradient if they work properly.

```@docs
GeometricFlux.AbstractGraphLayer
```
