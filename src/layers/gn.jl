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

function aggregate_neighbors(::GraphNet, el::NamedTuple, aggr, E)
    batch_size = size(E)[end]
    dstsize = (size(E, 1), el.N, batch_size)
    xs = batched_index(el.xs, batch_size)
    return _scatter(aggr, E, xs, dstsize)
end

aggregate_neighbors(::GraphNet, el::NamedTuple, aggr, E::AbstractMatrix) = _scatter(aggr, E, el.xs)

@inline aggregate_neighbors(::GraphNet, ::NamedTuple, ::Nothing, E) = nothing
@inline aggregate_neighbors(::GraphNet, ::NamedTuple, ::Nothing, ::AbstractMatrix) = nothing

aggregate_edges(::GraphNet, aggr, E) = aggregate(aggr, E)
@inline aggregate_edges(::GraphNet, ::Nothing, E) = nothing

aggregate_vertices(::GraphNet, aggr, V) = aggregate(aggr, V)
@inline aggregate_vertices(::GraphNet, ::Nothing, V) = nothing

function propagate(gn::GraphNet, sg::SparseGraph, E, V, u, naggr, eaggr, vaggr)
    el = GraphSignals.to_namedtuple(sg)
    return propagate(gn, el, E, V, u, naggr, eaggr, vaggr)
end

"""
- `update_batch_edge`: (E_in_dim, E) -> (E_out_dim, E)
- `aggregate_neighbors`: (E_out_dim, E) -> (E_out_dim, V)
- `update_batch_vertex`: (V_in_dim, V) -> (V_out_dim, V)
- `aggregate_edges`: (E_out_dim, E) -> (E_out_dim, 1)
- `aggregate_vertices`: (V_out_dim, V) -> (V_out_dim, 1)
- `update_global`: (dim, 1) -> (dim, 1)
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

WithGraph(fg::AbstractFeaturedGraph, gn::GraphNet) =
    WithGraph(GraphSignals.to_namedtuple(fg), gn, positional_feature(fg))

WithGraph(gn::GraphNet; dynamic=nothing) =
    WithGraph(DynamicGraph(dynamic), gn, GraphSignals.NullDomain())

WithGraph(fg::AbstractFeaturedGraph, gn::GraphNet, pos::GraphSignals.AbstractGraphDomain) =
    WithGraph(GraphSignals.to_namedtuple(fg), gn, pos)
