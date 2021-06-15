using LightGraphs: AbstractGraph
using MetaGraphs: AbstractMetaGraph

## Convolution layers accepting AbstractGraph

GCNConv(g::AbstractGraph, ch::Pair{<:Integer,<:Integer}, σ = identity; kwargs...) =
    GCNConv(FeaturedGraph(g), ch, σ; kwargs...)

ChebConv(g::AbstractGraph, ch::Pair{<:Integer,<:Integer}, k::Integer; kwargs...) =
    ChebConv(FeaturedGraph(g), ch, k; kwargs...)

GraphConv(g::AbstractGraph, ch::Pair{<:Integer,<:Integer}, σ=identity, aggr=+; kwargs...) =
    GraphConv(FeaturedGraph(g), ch, σ, aggr; kwargs...)

GATConv(g::AbstractGraph, ch::Pair{<:Integer,<:Integer}; kwargs...) =
    GATConv(FeaturedGraph(g), ch; kwargs...)

GatedGraphConv(g::AbstractGraph, out_ch::Integer, num_layers::Integer; kwargs...) =
    GatedGraphConv(FeaturedGraph(g), out_ch, num_layers; kwargs...)

EdgeConv(g::AbstractGraph, nn; kwargs...) = EdgeConv(FeaturedGraph(g), nn; kwargs...)


## Convolution layers accepting AbstractMetaGraph

GCNConv(g::AbstractMetaGraph, ch::Pair{<:Integer,<:Integer}, σ=identity; kwargs...) =
    GCNConv(g.graph, ch, σ; kwargs...)

ChebConv(g::AbstractMetaGraph, ch::Pair{<:Integer,<:Integer}, k::Integer; kwargs...) =
    ChebConv(g.graph, ch, k; kwargs...)

GraphConv(g::AbstractMetaGraph, ch::Pair{<:Integer,<:Integer}, σ=identity, aggr=+; kwargs...) =
    GraphConv(g.graph, ch, σ, aggr; kwargs...)

GATConv(g::AbstractMetaGraph, ch::Pair{<:Integer,<:Integer}; kwargs...) =
    GATConv(g.graph, ch; kwargs...)

GatedGraphConv(g::AbstractMetaGraph, out_ch::Integer, num_layers::Integer; kwargs...) =
    GatedGraphConv(g.graph, out_ch, num_layers; kwargs...)

EdgeConv(g::AbstractMetaGraph, nn; kwargs...) = EdgeConv(g.graph, nn; kwargs...)
