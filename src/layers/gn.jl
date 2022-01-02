_view(::Nothing, idx) = nothing
_view(A::Fill{T,2,Axes}, idx) where {T,Axes} = fill(A.value, A.axes[1], length(idx))

function _view(A::SubArray{T,2,S}, idx) where {T,S<:Fill}
    p = parent(A)
    return Fill(p.value, p.axes[1].stop, length(idx))
end

_view(A::AbstractMatrix, idx) = view(A, :, idx)

function _view(A::SubArray{T,2,S}, idxs) where {T,S<:AbstractMatrix}
    view_idx = A.indices[2]
    if view_idx == idxs
        return A
    else
        idxs = findall(x -> x in idxs, view_idx)
        return view(A, :, idxs)
    end
end

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

@inline function update_batch_edge(gn::GraphNet, fg::AbstractFeaturedGraph, E, V, u)
    es = Zygote.ignore(()->cpu(GraphSignals.incident_edges(fg)))
    xs = Zygote.ignore(()->cpu(GraphSignals.repeat_nodes(fg)))
    nbrs = Zygote.ignore(()->cpu(GraphSignals.neighbors(fg)))
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
    N = nv(parent(graph(fg).S))
    xs = Zygote.ignore(()->cpu(GraphSignals.repeat_nodes(fg)))
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
    FeaturedGraph(fg, nf=V, ef=E, gf=u)
end

function propagate(gn::GraphNet, fg::AbstractFeaturedGraph, E::AbstractArray, V::AbstractArray, u::AbstractArray,
                   naggr=nothing, eaggr=nothing, vaggr=nothing)
    E = update_batch_edge(gn, fg, E, V, u)
    Ē = aggregate_neighbors(gn, fg, naggr, E)
    V = update_batch_vertex(gn, fg, Ē, V, u)
    ē = aggregate_edges(gn, eaggr, E)
    v̄ = aggregate_vertices(gn, vaggr, V)
    u = update_global(gn, ē, v̄, u)
    return parent(E), parent(V), u
end
