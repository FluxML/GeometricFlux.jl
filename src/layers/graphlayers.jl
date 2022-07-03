"""
    AbstractGraphLayer

An abstract type of graph neural network layer for GeometricFlux.
"""
abstract type AbstractGraphLayer end

"""
    WithGraph([g], layer; dynamic=nothing)

Train GNN layers with static graph.

# Arguments

- `g`: If a `FeaturedGraph` is given, a fixed graph is used to train with.
- `layer`: A GNN layer.
- `dynamic`: If a function is given, it enables dynamic graph update by constructing
dynamic graph through given function within layers.

# Example

```jldoctest
julia> using GraphSignals, GeometricFlux

julia> adj = [0 1 0 1;
              1 0 1 0;
              0 1 0 1;
              1 0 1 0];

julia> fg = FeaturedGraph(adj);

julia> gc = WithGraph(fg, GCNConv(1024=>256))  # graph preprocessed by adding self loops
WithGraph(Graph(#V=4, #E=8), GCNConv(1024 => 256))

julia> WithGraph(fg, Dense(10, 5))
Dense(10 => 5)      # 55 parameters

julia> model = Chain(
           GCNConv(32=>32),
           gc,
       );

julia> WithGraph(fg, model)
Chain(
  WithGraph(
    GCNConv(32 => 32),                  # 1_056 parameters
  ),
  WithGraph(
    GCNConv(1024 => 256),               # 262_400 parameters
  ),
)         # Total: 4 trainable arrays, 263_456 parameters,
          # plus 2 non-trainable, 32 parameters, summarysize 1.006 MiB.
```
"""
struct WithGraph{L<:AbstractGraphLayer,G,P}
    graph::G
    layer::L
    position::P
end

@functor WithGraph

Flux.trainable(l::WithGraph) = (l.layer, )

function Optimisers.destructure(m::WithGraph)
    p, re = destructure(m.layer)
    function  re_withgraph(x)
        WithGraph(re(x), m.fg)        
    end

    return p, re_withgraph
end

function Base.show(io::IO, l::WithGraph)
    print(io, "WithGraph(Graph(#V=", nv(l.graph), ", #E=", ne(l.graph), "), ")
    print(io, l.layer)
    has_positional_feature(l.position) &&
        print(io, ", domain_dim=", GraphSignals.pf_dims_repr(l.position))
    print(io, ")")
end

WithGraph(fg::AbstractFeaturedGraph, model::Chain; kwargs...) =
    Chain([WithGraph(fg, l; kwargs...) for l in model.layers]...)
WithGraph(::AbstractFeaturedGraph, layer::WithGraph; kwargs...) = layer
WithGraph(::AbstractFeaturedGraph, layer; kwargs...) = layer

update_batch_edge(l::WithGraph, args...) = update_batch_edge(l.layer, l.graph, args...)
aggregate_neighbors(l::WithGraph, args...) = aggregate_neighbors(l.layer, l.graph, args...)
update_batch_vertex(l::WithGraph, args...) = update_batch_vertex(l.layer, l.graph, args...)

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
julia> using Flux, GeometricFlux

julia> l = GraphParallel(
            node_layer=Dropout(0.5),
            global_layer=Dense(10, 5)
       )
GraphParallel(node_layer=Dropout(0.5), edge_layer=identity, global_layer=Dense(10 => 5))
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

function (l::GraphParallel)(fg::AbstractFeaturedGraph)
    nf = l.node_layer(node_feature(fg))
    ef = l.edge_layer(edge_feature(fg))
    gf = l.global_layer(global_feature(fg))
    return ConcreteFeaturedGraph(fg, nf=nf, ef=ef, gf=gf)
end

function Base.show(io::IO, l::GraphParallel)
    print(io, "GraphParallel(")
    print(io, "node_layer=", l.node_layer)
    print(io, ", edge_layer=", l.edge_layer)
    print(io, ", global_layer=", l.global_layer)
    print(io, ")")
end

struct DynamicGraph{F}
    method::F
end
