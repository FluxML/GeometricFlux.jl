abstract type GraphNet <: AbstractGraphLayer end

@inline update_edge(gn::GraphNet, e, vi, vj, u) = e
@inline update_vertex(gn::GraphNet, ē, vi, u) = vi
@inline update_global(gn::GraphNet, ē, v̄, u) = u

@inline function update_batch_edge(gn::GraphNet, fg::AbstractFeaturedGraph, E, V, u)
    es, xs, nbrs = all_edges(fg)
    return update_edge(
        gn,
        batched_gather(E, es),
        batched_gather(V, xs),
        batched_gather(V, nbrs),
        u
    )
end

@inline function update_batch_vertex(gn::GraphNet, fg::AbstractFeaturedGraph, Ē, V, u)
    # nodes = Zygote.ignore(()->vertices(fg))
    # return update_vertex(gn, _gather(Ē, nodes), _gather(V, nodes), u)
    return update_vertex(gn, Ē, V, u)
end

@inline function aggregate_neighbors(gn::GraphNet, fg::AbstractFeaturedGraph, aggr, E)
    N = nv(parent(fg))
    batch_size = size(E)[end]
    es, xs, nbrs = all_edges(fg)
    xs = batched_index(xs, batch_size)
    Ē = NNlib.scatter(aggr, E, xs; dstsize=(size(E, 1), N, batch_size))
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
