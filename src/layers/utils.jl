"""
    WithGraph(layer, fg, [subgraph=:])

Train GNN layers with fixed graph.

# Arguments

- `layer`: A GNN layer.
- `fg`: A fixed `FeaturedGraph` to train with.
- `subgraph`: Node indeices to get a subgraph from `fg`.

# Example

```jldoctest
julia> adj = [0 1 0 1;
              1 0 1 0;
              0 1 0 1;
              1 0 1 0];

julia> fg = FeaturedGraph(adj);

julia> gc = WithGraph(GCNConv(1024=>256), fg)
WithGraph(GCNConv(1024 => 256), FeaturedGraph(#V=4, #E=4))

julia> subgraph = [1, 2, 4]  # specify subgraph nodes

julia> gc = WithGraph(GCNConv(1024=>256), fg, subgraph)
WithGraph(GCNConv(1024 => 256), FeaturedGraph(#V=4, #E=4), subgraph=[1, 2, 4])
```
"""
struct WithGraph{L,G<:AbstractFeaturedGraph,S}
    layer::L
    fg::G
    subgraph::S
end

@functor WithGraph

Flux.trainable(l::WithGraph) = (l.layer, )

WithGraph(layer, fg::AbstractFeaturedGraph) = WithGraph(layer, fg, :)

function Base.show(io::IO, l::WithGraph)
    print(io, "WithGraph(")
    print(io, l.layer, ", ")
    print(io, "FeaturedGraph(#V=", nv(l.fg), ", #E=", ne(l.fg), ")")
    l.subgraph == (:) || print(io, ", subgraph=", l.subgraph)
    print(io, ")")
end

"""
    GraphParallel(; node_layer=identity, edge_layer=identity, global_layer=identity)

Passing features in `FeaturedGraph` in parallel. It takes `FeaturedGraph` as input
and it can be specified by assigning layers for specific (node, edge and global) features.

# Arguments

- `node_layer`: A regular Flux layer for passing node features.
- `edge_layer`: A regular Flux layer for passing edge features.
- `global_layer`: A regular Flux layer for passing global features.

# Example

```jldoctest
julia> l = GraphParallel(
            node_layer=Dropout(0.5),
            global_layer=Dense(10, 5)
       )
```
"""
struct GraphParallel{N,E,G}
    node_layer::N
    edge_layer::E
    global_layer::G
end

@functor GraphParallel

GraphParallel(; node_layer=identity, edge_layer=identity, global_layer=identity) =
    GraphParallel(node_layer, edge_layer, global_layer)

function (l::GraphParallel)(fg::FeaturedGraph)
    nf = l.node_layer(node_feature(fg))
    ef = l.edge_layer(edge_feature(fg))
    gf = l.global_layer(global_feature(fg))
    return FeaturedGraph(fg, nf=nf, ef=ef, gf=gf)
end

function Base.show(io::IO, l::GraphParallel)
    print(io, "GraphParallel(")
    print(io, "node_layer=", l.node_layer)
    print(io, ", edge_layer=", l.edge_layer)
    print(io, ", global_layer=", l.global_layer)
    print(io, ")")
end
