using MetaGraphs: AbstractMetaGraph

GCNConv(g::AbstractMetaGraph, ch::Pair{<:Integer,<:Integer}, σ=identity; kwargs...) =
    GCNConv(g.graph, ch, σ; kwargs...)


ChebConv(g::AbstractMetaGraph, ch::Pair{<:Integer,<:Integer}, k::Integer; kwargs...) =
    ChebConv(g.graph, ch, k; kwargs...)


GraphConv(g::AbstractMetaGraph, ch::Pair{<:Integer,<:Integer}, aggr=:add; kwargs...) =
    GraphConv(g.graph, ch, aggr; kwargs...)


GATConv(g::AbstractMetaGraph, ch::Pair{<:Integer,<:Integer}; kwargs...) =
    GATConv(g.graph, ch; kwargs...)


GatedGraphConv(g::AbstractMetaGraph, out_ch::Integer, num_layers::Integer; kwargs...) =
    GatedGraphConv(g.graph, out_ch, num_layers; kwargs...)


EdgeConv(g::AbstractMetaGraph, nn; kwargs...) = EdgeConv(g.graph, nn; kwargs...)
