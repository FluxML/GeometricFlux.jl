abstract type GraphNet <: AbstractGraphLayer end

update_edge(::GraphNet, e, vi, vj, u) = e
update_vertex(::GraphNet, ē, vi, u) = vi
update_global(::GraphNet, ē, v̄, u) = u

update_batch_edge(gn::GraphNet, el::NamedTuple, E, V, u) =
    update_edge(
        gn,
        _gather(E, el.es),
        _gather(V, el.xs),
        _gather(V, el.nbrs),
        u
    )

update_batch_vertex(gn::GraphNet, ::NamedTuple, Ē, V, u) = update_vertex(gn, Ē, V, u)

function aggregate_neighbors(gn::GraphNet, el::NamedTuple, aggr, E)
    batch_size = size(E)[end]
    dstsize = (size(E, 1), el.N, batch_size)
    xs = batched_index(el.xs, batch_size)
    return aggregate_neighbors(gn, xs, aggr, E, dstsize)
end

function aggregate_neighbors(gn::GraphNet, el::NamedTuple, aggr, E::AbstractMatrix)
    return aggregate_neighbors(gn, el.xs, aggr, E)
end

@inline aggregate_neighbors(::GraphNet, ::NamedTuple, ::Nothing, E) = nothing

function aggregate_neighbors(::GraphNet, xs::AbstractArray, aggr, E, dstsize=nothing)
    if isnothing(dstsize)
        return NNlib.scatter(aggr, E, xs)
    else
        return NNlib.scatter(aggr, E, xs; dstsize=dstsize)
    end
end

aggregate_edges(::GraphNet, aggr, E) = aggregate(aggr, E)
@inline aggregate_edges(::GraphNet, ::Nothing, E) = nothing

aggregate_vertices(::GraphNet, aggr, V) = aggregate(aggr, V)
@inline aggregate_vertices(::GraphNet, ::Nothing, V) = nothing

function propagate(gn::GraphNet, sg::SparseGraph, E, V, u, naggr, eaggr, vaggr)
    es, nbrs, xs = Zygote.ignore(() -> collect(edges(sg)))
    el = (N=nv(sg), E=ne(sg), es=es, nbrs=nbrs, xs=xs)
    return propagate(gn, el, E, V, u, naggr, eaggr, vaggr)
end

"""
- `update_batch_edge`: (E_in_dim, E) -> (E_out_dim, E)
- `aggregate_neighbors`: (E_out_dim, E) -> (E_out_dim, V)
- `update_batch_vertex`: (V_in_dim, V) -> (V_out_dim, V)
- `aggregate_edges`: (E_out_dim, E) -> (E_out_dim,)
- `aggregate_vertices`: (V_out_dim, V) -> (V_out_dim,)
- `update_global`: (dim,) -> (dim,)
"""
function propagate(gn::GraphNet, el::NamedTuple, E, V, u, naggr, eaggr, vaggr)
    E = update_batch_edge(gn, el, E, V, u)
    Ē = aggregate_neighbors(gn, el, naggr, E)
    V = update_batch_vertex(gn, el, Ē, V, u)
    ē = aggregate_edges(gn, eaggr, E)
    v̄ = aggregate_vertices(gn, vaggr, V)
    u = update_global(gn, ē, v̄, u)
    return E, V, u
end
