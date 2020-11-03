abstract type MessagePassing <: GraphNet end

"""
    message(mp, x_i, x_j, e_ij)

Message function for message-passing scheme. This function can be overrided to dispatch to custom layers.
First argument should be message-passing layer, the rest of arguments can be `x_i`, `x_j` and `e_ij`.

# Arguments
- `mp`: message-passing layer.
- `x_i`: the feature of node `x_i`.
- `x_j`: the feature of neighbors of node `x_i`.
- `e_ij`: the feature of edge (`x_i`, `x_j`).
"""
@inline message(mp::T, x_i, x_j, e_ij) where {T<:MessagePassing} = x_j
@inline message(mp::T, i::Integer, j::Integer, x_i, x_j, e_ij) where {T<:MessagePassing} = x_j

"""
    update(mp, m, x)

Update function for message-passing scheme. This function can be overrided to dispatch to custom layers.
First argument should be message-passing layer, the rest of arguments can be `X` and `M`.

# Arguments
- `mp`: message-passing layer.
- `m`: the message aggregated from message function.
- `x`: the single node feature.
"""
@inline update(mp::T, m, x) where {T<:MessagePassing} = m
@inline update(mp::T, i::Integer, m, x) where {T<:MessagePassing} = m

@inline function update_batch_edge(mp::T, adj, E::AbstractMatrix, X::AbstractMatrix) where {T<:MessagePassing}
    n = size(adj, 1)
    edge_idx = edge_index_table(adj)
    hcat([apply_batch_message(mp, i, adj[i], edge_idx, E, X) for i in 1:n]...)
end

@inline function apply_batch_message(mp::T, i, js, edge_idx, E::AbstractMatrix, X::AbstractMatrix) where {T<:MessagePassing}
    hcat([message(mp, get_feature(X, i), get_feature(X, j), get_feature(E, edge_idx[(i,j)])) for j = js]...)
end

@inline function update_batch_vertex(mp::T, M::AbstractMatrix, X::AbstractMatrix) where {T<:MessagePassing}
    hcat([update(mp, get_feature(M, i), get_feature(X, i)) for i in 1:size(X,2)]...)
end

@inline function aggregate_neighbors(mp::T, aggr::Symbol, M::AbstractMatrix, accu_edge) where {T<:MessagePassing}
    @assert !iszero(accu_edge) "accumulated edge must not be zero."
    cluster = generate_cluster(M, accu_edge)
    pool(aggr, cluster, M)
end

function propagate(mp::T, fg::FeaturedGraph, aggr::Symbol=:add) where {T<:MessagePassing}
    E, X = propagate(mp, adjacency_list(fg), fg.ef, fg.nf, aggr)
    FeaturedGraph(graph(fg), X, E, zeros(0))
end

function propagate(mp::T, adj::AbstractVector{S}, E::R, X::Q, aggr::Symbol) where {T<:MessagePassing,S<:AbstractVector,R,Q}
    accu_edge = accumulated_edges(adj)

    E = update_batch_edge(mp, adj, E, X)
    M = aggregate_neighbors(mp, aggr, E, accu_edge)
    X = update_batch_vertex(mp, M, X)

    E, X
end
