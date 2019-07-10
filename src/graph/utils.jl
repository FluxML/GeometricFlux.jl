using LightGraphs: AbstractSimpleGraph
using SimpleWeightedGraphs: AbstractSimpleWeightedGraph, outneighbors

function adjlist(g::AbstractSimpleGraph)
    N = nv(g)
    el = Vector{Int}[outneighbors(g, i) for i = 1:N]
    return el
end

function adjlist(g::AbstractSimpleWeightedGraph)
    N = nv(g)
    el = Vector{Int}[outneighbors(g, i) for i = 1:N]
    return el
end
