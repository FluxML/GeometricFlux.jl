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

@inline function update_batch_edge(mp::T, adj, E::AbstractMatrix, X::AbstractMatrix, u) where {T<:MessagePassing}
    n = size(adj, 1)
    edge_idx = edge_index_table(adj)
    mapreduce(i -> apply_batch_message(mp, i, adj[i], edge_idx, E, X, u), hcat, 1:n)
end

@inline apply_batch_message(mp::T, i, js, edge_idx, E::AbstractMatrix, X::AbstractMatrix, u) where {T<:MessagePassing} =
    mapreduce(j -> message(mp, get_feature(X, i), get_feature(X, j), get_feature(E, edge_idx[(i,j)])), hcat, js)

@inline update_batch_vertex(mp::T, M::AbstractMatrix, X::AbstractMatrix, u) where {T<:MessagePassing} = 
    mapreduce(i -> update(mp, get_feature(M, i), get_feature(X, i)), hcat, 1:size(X,2))

@inline function aggregate_neighbors(mp::T, aggr, M::AbstractMatrix, accu_edge) where {T<:MessagePassing}
    @assert !iszero(accu_edge) "accumulated edge must not be zero."
    cluster = generate_cluster(M, accu_edge)
    GeometricFlux.scatter(aggr, cluster, M)
end

function propagate(mp::T, fg::FeaturedGraph, aggr=+) where {T<:MessagePassing}
    E, X = propagate(mp, adjacency_list(fg), fg.ef, fg.nf, aggr)
    FeaturedGraph(graph(fg), nf=X, ef=E, gf=Fill(0.f0, 0))
end

function propagate(mp::T, adj::AbstractVector{S}, E::R, X::Q, aggr) where {T<:MessagePassing,S<:AbstractVector,R,Q}
    E, X, u = propagate(mp, adj, E, X, Fill(0.f0, 0), aggr, nothing, nothing)
    E, X
end
