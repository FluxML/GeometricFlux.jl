abstract type GraphNet <: AbstractGraphLayer end

"""
    update_edge(gn, e, vi, vj, u)

Update function for edge feature in graph network.

# Arguments

- `gn::GraphNet`: A graph network layer.
- `e`: Edge feature.
- `vi`: Node feature for node `i`.
- `vj`: Node feature for neighbors of node `i`.
- `u`: Global feature.

See also [`update_vertex`](@ref), [`update_global`](@ref), [`update_batch_edge`](@ref),
[`update_batch_vertex`](@ref), [`aggregate_neighbors`](@ref), [`aggregate_edges`](@ref),
[`aggregate_vertices`](@ref).
"""
update_edge(::GraphNet, e, vi, vj, u) = e

"""
    update_vertex(gn, ē, vi, u)

Update function for node feature in graph network.

# Arguments

- `gn::GraphNet`: A graph network layer.
- `ē`: Aggregated edge feature.
- `vi`: Node feature for node `i`.
- `u`: Global feature.

See also [`update_edge`](@ref), [`update_global`](@ref), [`update_batch_edge`](@ref),
[`update_batch_vertex`](@ref), [`aggregate_neighbors`](@ref), [`aggregate_edges`](@ref),
[`aggregate_vertices`](@ref).
"""
update_vertex(::GraphNet, ē, vi, u) = vi

"""
    update_global(gn, ē, v̄, u)

Update function for global feature in graph network.

# Arguments

- `gn::GraphNet`: A graph network layer.
- `ē`: Aggregated edge feature.
- `v̄`: Aggregated node feature for node `i`.
- `u`: Global feature.

See also [`update_edge`](@ref), [`update_vertex`](@ref), [`update_batch_edge`](@ref),
[`update_batch_vertex`](@ref), [`aggregate_neighbors`](@ref), [`aggregate_edges`](@ref),
[`aggregate_vertices`](@ref).
"""
update_global(::GraphNet, ē, v̄, u) = u

"""
    update_batch_edge(gn, el, E, V, u)

Returns new edge features of size `(E_out_dim, #E, [batch_size])`.

# Arguments

- `gn::GraphNet`: A graph network layer.
- `el::NamedTuple`: Collection of graph information.
- `E`: All edge features. Its size should be `(E_in_dim, #E, [batch_size])`.
- `V`: All node features.
- `u`: Global features.

See also [`update_edge`](@ref), [`update_vertex`](@ref), [`update_global`](@ref),
[`update_batch_vertex`](@ref), [`aggregate_neighbors`](@ref), [`aggregate_edges`](@ref),
[`aggregate_vertices`](@ref).
"""
update_batch_edge(gn::GraphNet, el::NamedTuple, E, V, u) =
    update_edge(
        gn,
        _gather(E, el.es),
        _gather(V, el.xs),
        _gather(V, el.nbrs),
        u
    )

"""
    update_batch_vertex(gn, el, Ē, V, u)

Returns new node features of size `(V_out_dim, #V, [batch_size])`.

# Arguments

- `gn::GraphNet`: A graph network layer.
- `el::NamedTuple`: Collection of graph information.
- `Ē`: All edge features. Its size should be `(E_in_dim, #V, [batch_size])`.
- `V`: All node features. Its size should be `(V_in_dim, #V, [batch_size])`.
- `u`: Global features.

See also [`update_edge`](@ref), [`update_vertex`](@ref), [`update_global`](@ref),
[`update_batch_edge`](@ref), [`aggregate_neighbors`](@ref), [`aggregate_edges`](@ref),
[`aggregate_vertices`](@ref).
"""
update_batch_vertex(gn::GraphNet, ::NamedTuple, Ē, V, u) = update_vertex(gn, Ē, V, u)

"""
    aggregate_neighbors(gn, el, aggr, E)

Returns aggregated neighbor features of size `(E_out_dim, #V, [batch_size])`.

# Arguments

- `gn::GraphNet`: A graph network layer.
- `el::NamedTuple`: Collection of graph information.
- `aggr`: Aggregate function to apply on neighbor features.
- `E`: All edge features from neighbors. Its size should be `(E_out_dim, #E, [batch_size])`.

See also [`update_edge`](@ref), [`update_vertex`](@ref), [`update_global`](@ref),
[`update_batch_edge`](@ref), [`update_batch_vertex`](@ref), [`aggregate_edges`](@ref),
[`aggregate_vertices`](@ref).
"""
function aggregate_neighbors(::GraphNet, el::NamedTuple, aggr, E)
    batch_size = size(E)[end]
    dstsize = (size(E, 1), el.N, batch_size)
    xs = batched_index(el.xs, batch_size)
    return _scatter(aggr, E, xs, dstsize)
end

aggregate_neighbors(::GraphNet, el::NamedTuple, aggr, E::AbstractMatrix) = _scatter(aggr, E, el.xs)

@inline aggregate_neighbors(::GraphNet, ::NamedTuple, ::Nothing, E) = nothing
@inline aggregate_neighbors(::GraphNet, ::NamedTuple, ::Nothing, ::AbstractMatrix) = nothing

"""
    aggregate_edges(gn, aggr, E)

Returns aggregated edge features of size `(E_out_dim, 1, [batch_size])` for updating global feature.

# Arguments

- `gn::GraphNet`: A graph network layer.
- `aggr`: Aggregate function to apply on edge features.
- `E`: All edge features. Its size should be `(E_out_dim, #E, [batch_size])`.

See also [`update_edge`](@ref), [`update_vertex`](@ref), [`update_global`](@ref),
[`update_batch_edge`](@ref), [`update_batch_vertex`](@ref), [`aggregate_neighbors`](@ref),
[`aggregate_vertices`](@ref).
"""
aggregate_edges(::GraphNet, aggr, E) = aggregate(aggr, E)
@inline aggregate_edges(::GraphNet, ::Nothing, E) = nothing

"""
    aggregate_vertices(gn, aggr, V)

Returns aggregated node features of size `(V_out_dim, 1, [batch_size])` for updating global feature.

# Arguments

- `gn::GraphNet`: A graph network layer.
- `aggr`: Aggregate function to apply on node features.
- `V`: All node features. Its size should be `(V_out_dim, #V, [batch_size])`.

See also [`update_edge`](@ref), [`update_vertex`](@ref), [`update_global`](@ref),
[`update_batch_edge`](@ref), [`update_batch_vertex`](@ref), [`aggregate_neighbors`](@ref),
[`aggregate_edges`](@ref).
"""
aggregate_vertices(::GraphNet, aggr, V) = aggregate(aggr, V)
@inline aggregate_vertices(::GraphNet, ::Nothing, V) = nothing

function propagate(gn::GraphNet, sg::SparseGraph, E, V, u, naggr, eaggr, vaggr)
    el = GraphSignals.to_namedtuple(sg)
    return propagate(gn, el, E, V, u, naggr, eaggr, vaggr)
end

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
