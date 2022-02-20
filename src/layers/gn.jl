_view(::Nothing, idx) = nothing
_view(A::Fill{T,2,Axes}, idx) where {T,Axes} = fill(A.value, A.axes[1], length(idx))
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

function _get_indices(fg::AbstractFeaturedGraph)
    es = cpu(GraphSignals.incident_edges(fg))
    xs = cpu(GraphSignals.repeat_nodes(fg))
    nbrs = cpu(GraphSignals.neighbors(fg))
    sorted_idx = sort!(collect(zip(es, xs, nbrs)), by=x->x[1])
    return collect.(collect(zip(sorted_idx...)))
end

@inline function update_batch_edge(gn::GraphNet, fg::AbstractFeaturedGraph, E, V, u)
    es, xs, nbrs = Zygote.ignore(()->_get_indices(fg))
    ms = map((e,i,j)->update_edge(gn, _view(E, e), _view(V, i), _view(V, j), u), es, xs, nbrs)
    M = hcat_by_sum(ms)
    return M
end

@inline function update_batch_vertex(gn::GraphNet, fg::AbstractFeaturedGraph, Ē, V, u)
    nodes = Zygote.ignore(()->vertices(fg))
    vs = map(n->update_vertex(gn, _view(Ē, n), _view(V, n), u), nodes)
    V_ = hcat_by_sum(vs)
    return V_
end

@inline function aggregate_neighbors(gn::GraphNet, fg::AbstractFeaturedGraph, aggr, E)
    N = nv(parent(fg))
    es, xs, nbrs = Zygote.ignore(()->_get_indices(fg))
    Ē = NNlib.scatter(aggr, E, xs; dstsize=(size(E, 1), N))
    return Ē
end
@inline aggregate_neighbors(gn::GraphNet, fg::AbstractFeaturedGraph, aggr::Nothing, @nospecialize E) = nothing

@inline aggregate_edges(gn::GraphNet, aggr, E) = aggregate(aggr, E)
@inline aggregate_edges(gn::GraphNet, aggr::Nothing, @nospecialize E) = nothing

@inline aggregate_vertices(gn::GraphNet, aggr, V) = aggregate(aggr, V)
@inline aggregate_vertices(gn::GraphNet, aggr::Nothing, @nospecialize V) = nothing

function propagate(gn::GraphNet, fg::AbstractFeaturedGraph, naggr=nothing, eaggr=nothing, vaggr=nothing)
    E, V, u = propagate(gn, fg, edge_feature(fg), node_feature(fg), global_feature(fg), naggr, eaggr, vaggr)
    return FeaturedGraph(fg, nf=V, ef=E, gf=u)
end

"""
- `update_batch_edge`: (E_in_dim, E) -> (E_out_dim, E)
- `aggregate_neighbors`: (E_out_dim, E) -> (E_out_dim, V)
- `update_batch_vertex`: (V_in_dim, V) -> (V_out_dim, V)
- `aggregate_edges`: (E_out_dim, E) -> (E_out_dim,)
- `aggregate_vertices`: (V_out_dim, V) -> (V_out_dim,)
- `update_global`: (dim,) -> (dim,)
"""
function propagate(gn::GraphNet, fg::AbstractFeaturedGraph, E::AbstractArray, V::AbstractArray, u::AbstractArray,
                   naggr=nothing, eaggr=nothing, vaggr=nothing)
    E = update_batch_edge(gn, fg, E, V, u)
    Ē = aggregate_neighbors(gn, fg, naggr, E)
    V = update_batch_vertex(gn, fg, Ē, V, u)
    ē = aggregate_edges(gn, eaggr, E)
    v̄ = aggregate_vertices(gn, vaggr, V)
    u = update_global(gn, ē, v̄, u)
    return E, V, u
end
