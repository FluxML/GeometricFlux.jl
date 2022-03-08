abstract type GraphNet <: AbstractGraphLayer end

update_edge(gn::GraphNet, e, vi, vj, u) = e
update_vertex(gn::GraphNet, ē, vi, u) = vi
update_global(gn::GraphNet, ē, v̄, u) = u

update_batch_edge(gn::GraphNet, sg::SparseGraph, E, V, u) =
    update_edge(gn, collect(edges(sg)), E, V, u)

update_batch_edge(gn::GraphNet, el::NamedTuple, E, V, u) =
    update_edge(
        gn,
        batched_gather(E, el.es),
        batched_gather(V, el.xs),
        batched_gather(V, el.nbrs),
        u
    )


update_batch_vertex(gn::GraphNet, sg::SparseGraph, Ē, V, u) =
    update_vertex(gn, Ē, V, u)

update_batch_vertex(gn::GraphNet, el::NamedTuple, Ē, V, u) =
    update_vertex(gn, Ē, V, u)

function aggregate_neighbors(gn::GraphNet, sg::SparseGraph, aggr, E)
    N = nv(sg)
    batch_size = size(E)[end]
    dstsize = (size(E, 1), N, batch_size)
    _, _, xs = collect(edges(sg))
    xs = batched_index(xs, batch_size)
    return aggregate_neighbors(gn, xs, dstsize, aggr, E)
end

@inline aggregate_neighbors(gn::GraphNet, sg::SparseGraph, aggr::Nothing, @nospecialize E) = nothing

function aggregate_neighbors(gn::GraphNet, el::NamedTuple, aggr, E)
    batch_size = size(E)[end]
    dstsize = (size(E, 1), el.N, batch_size)
    xs = batched_index(el.xs, batch_size)
    return aggregate_neighbors(gn, xs, dstsize, aggr, E)
end

function aggregate_neighbors(gn::GraphNet, xs::AbstractArray, dstsize, aggr, E)
    return NNlib.scatter(aggr, E, xs; dstsize=dstsize)
end

@inline aggregate_edges(gn::GraphNet, aggr, E) = aggregate(aggr, E)
@inline aggregate_edges(gn::GraphNet, aggr::Nothing, @nospecialize E) = nothing

@inline aggregate_vertices(gn::GraphNet, aggr, V) = aggregate(aggr, V)
@inline aggregate_vertices(gn::GraphNet, aggr::Nothing, @nospecialize V) = nothing

function propagate(gn::GraphNet, fg::AbstractFeaturedGraph, naggr=nothing, eaggr=nothing, vaggr=nothing)
    E, V, u = propagate(gn, graph(fg), edge_feature(fg), node_feature(fg), global_feature(fg), naggr, eaggr, vaggr)
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
function propagate(gn::GraphNet, sg::SparseGraph, E, V, u, naggr, eaggr, vaggr)
    E = update_batch_edge(gn, sg, E, V, u)
    Ē = aggregate_neighbors(gn, sg, naggr, E)
    V = update_batch_vertex(gn, sg, Ē, V, u)
    ē = aggregate_edges(gn, eaggr, E)
    v̄ = aggregate_vertices(gn, vaggr, V)
    u = update_global(gn, ē, v̄, u)
    return E, V, u
end
