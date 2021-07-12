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
@inline message(mp::MessagePassing, x_i, x_j, e_ij) = x_j
@inline message(mp::MessagePassing, i::Integer, j::Integer, x_i, x_j, e_ij) = x_j

"""
    update(mp, m, x)

Update function for message-passing scheme. This function can be overrided to dispatch to custom layers.
First argument should be message-passing layer, the rest of arguments can be `X` and `M`.

# Arguments
- `mp`: message-passing layer.
- `m`: the message aggregated from message function.
- `x`: the single node feature.
"""
@inline update(mp::MessagePassing, m, x) = m
@inline update(mp::MessagePassing, i::Integer, m, x) = m

@inline function update_batch_edge(mp::MessagePassing, adj, E::AbstractMatrix, X::AbstractMatrix, u)
    n = size(adj, 1)
    edge_idx = edge_index_table(adj)
    mapreduce(i -> apply_batch_message(mp, i, adj[i], edge_idx, E, X, u), hcat, 1:n)
end

@inline apply_batch_message(mp::MessagePassing, i, js, edge_idx, E::AbstractMatrix, X::AbstractMatrix, u) =
    mapreduce(j -> GeometricFlux.message(mp, _view(X, i), _view(X, j), _view(E, edge_idx[(i,j)])), hcat, js)

@inline update_batch_vertex(mp::MessagePassing, M::AbstractMatrix, X::AbstractMatrix, u) = 
    mapreduce(i -> GeometricFlux.update(mp, _view(M, i), _view(X, i)), hcat, 1:size(X,2))

@inline function aggregate_neighbors(mp::MessagePassing, aggr, M::AbstractMatrix, accu_edge)
    @assert !iszero(accu_edge) "accumulated edge must not be zero."
    cluster = generate_cluster(M, accu_edge)
    NNlib.scatter(aggr, M, cluster)
end

function propagate(mp::MessagePassing, fg::FeaturedGraph, aggr=+)
    E, X = propagate(mp, adjacency_list(fg), fg.ef, fg.nf, aggr)
    FeaturedGraph(graph(fg), nf=X, ef=E, gf=Fill(0.f0, 0))
end

function propagate(mp::MessagePassing, adj::AbstractVector{S}, E::R, X::Q, aggr) where {S<:AbstractVector,R,Q}
    E, X, u = propagate(mp, adj, E, X, Fill(0.f0, 0), aggr, nothing, nothing)
    E, X
end
