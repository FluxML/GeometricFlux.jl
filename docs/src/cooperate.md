# Cooperate with Flux Layers

GeometricFlux is designed to be compatible with Flux layers. Flux layers usually have array input and array output. Since the mechanism of "what you feed is what you get", the API for array type is compatible directly with other Flux layers. However, the API for [`FeaturedGraph`](@ref) is not compatible directly.

## Fetching Features from [`FeaturedGraph`](@ref) and Output Compatible Result with Flux Layers

With a layer outputs a [`FeaturedGraph`](@ref), it is not compatible with Flux layers. Since Flux layers need single feature in array form as input, node features, edge features and global features can be selected by using [`FeaturedGraph`](@ref) APIs: [`node_feature`](@ref), [`edge_feature`](@ref) or [`global_feature`](@ref), respectively.

```julia
model = Chain(
    GCNConv(1024=>256, relu),
    node_feature,  # or edge_feature or global_feature
    softmax
)
```

In a multitask learning scenario, multiple outputs are required. A branching selection of features can be made as follows:

```julia
model = Chain(
    GCNConv(1024=>256, relu),
    x -> (node_feature(x), global_feature(x)),
    (nf, gf) -> (softmax(nf), identity.(gf))
)
```

## Branching Different Features Through Different Layers

```@raw html
<figure>
    <img src="../assets/graphparallel.svg" width="70%" alt="graphparallel.svg" /><br>
    <figcaption><em>GraphParallel wraps regular Flux layers for different kinds of features for integration to GNN layers.</em></figcaption>
</figure>
```

A [`GraphParallel`](@ref) construct is designed for passing each feature through different layers from a [`FeaturedGraph`](@ref). An example is given as follow:

```julia
Flux.Chain(
    ...
    GraphParallel(
        node_layer=Dropout(0.5),
        edge_layer=Dense(1024, 256, relu),
        global_layer=identity,
    ),
    ...
)
```

[`GraphParallel`](@ref) will pass node feature to a `Dropout` layer and edge feature to a `Dense` layer. Meanwhile, a [`FeaturedGraph`](@ref) is decomposed and keep the graph in [`FeaturedGraph`](@ref) to the downstream layers. A new [`FeaturedGraph`](@ref) is constructed with processed node feature, edge feature and global feature. [`GraphParallel`](@ref) acts as a layer which accepts a [`FeaturedGraph`](@ref) and output a [`FeaturedGraph`](@ref). Thus, it by pass the graph in a [`FeaturedGraph`](@ref) but pass different features to different layers.

```@docs
GeometricFlux.GraphParallel
```
