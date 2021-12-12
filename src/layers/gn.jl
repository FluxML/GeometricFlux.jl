_view(::Nothing, i) = nothing
_view(A::Fill{T,2,Axes}, i) where {T,Axes} = view(A, :, 1)
_view(A::AbstractMatrix, idx) = view(A, :, idx)

aggregate(aggr::typeof(+), X) = vec(sum(X, dims=2))
aggregate(aggr::typeof(-), X) = -vec(sum(X, dims=2))
aggregate(aggr::typeof(*), X) = vec(prod(X, dims=2))
aggregate(aggr::typeof(/), X) = 1 ./ vec(prod(X, dims=2))
aggregate(aggr::typeof(max), X) = vec(maximum(X, dims=2))
aggregate(aggr::typeof(min), X) = vec(minimum(X, dims=2))
aggregate(aggr::typeof(mean), X) = vec(aggr(X, dims=2))

abstract type GraphNet <: AbstractGraphLayer end

@inline update_edge(gn::GraphNet, e, vi, vj, u) = e
@inline update_vertex(gn::GraphNet, ē, vi, u) = vi
@inline update_global(gn::GraphNet, ē, v̄, u) = u

@inline function update_batch_edge(gn::GraphNet, sg::SparseGraph, E, V, u)
    ys = map(i -> apply_batch_message(gn, sg, i, GraphSignals.cpu_neighbors(sg, i), E, V, u), vertices(sg))
    return hcat(ys...)
end

@inline function apply_batch_message(gn::GraphNet, sg::SparseGraph, i, js, E, V, u)
    # js still CuArray
    es = Zygote.ignore(() -> GraphSignals.cpu_incident_edges(sg, i))
    ys = map(k -> update_edge(gn, _view(E, es[k]), _view(V, i), _view(V, js[k]), u), 1:length(js))
    return hcat(ys...)
end

@inline function update_batch_vertex(gn::GraphNet, Ē, V, u)
    ys = map(i -> update_vertex(gn, _view(Ē, i), _view(V, i), u), 1:size(V,2))
    return hcat(ys...)
end

@inline aggregate_neighbors(gn::GraphNet, sg::SparseGraph, aggr, E) = neighbor_scatter(aggr, E, sg)
@inline aggregate_neighbors(gn::GraphNet, sg::SparseGraph, aggr::Nothing, @nospecialize E) = nothing

@inline aggregate_edges(gn::GraphNet, aggr, E) = aggregate(aggr, E)
@inline aggregate_edges(gn::GraphNet, aggr::Nothing, @nospecialize E) = nothing

@inline aggregate_vertices(gn::GraphNet, aggr, V) = aggregate(aggr, V)
@inline aggregate_vertices(gn::GraphNet, aggr::Nothing, @nospecialize V) = nothing

function propagate(gn::GraphNet, fg::FeaturedGraph, naggr=nothing, eaggr=nothing, vaggr=nothing)
    E, V, u = propagate(gn, graph(fg), edge_feature(fg), node_feature(fg), global_feature(fg), naggr, eaggr, vaggr)
    FeaturedGraph(fg, nf=V, ef=E, gf=u)
end

function propagate(gn::GraphNet, sg::SparseGraph, E, V, u, naggr=nothing, eaggr=nothing, vaggr=nothing)
    E = update_batch_edge(gn, sg, E, V, u)
    Ē = aggregate_neighbors(gn, sg, naggr, E)
    V = update_batch_vertex(gn, Ē, V, u)
    ē = aggregate_edges(gn, eaggr, E)
    v̄ = aggregate_vertices(gn, vaggr, V)
    u = update_global(gn, ē, v̄, u)
    return E, V, u
end
