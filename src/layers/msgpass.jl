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

@inline function update_batch_edge(mp::T, adj, E::AbstractMatrix, X::AbstractMatrix) where {T<:MessagePassing}
    edge_idx = edge_index_table(adj)
    E_ = Vector[]
    for (i, js) = enumerate(adj)
        for j = js
            k = edge_idx[(i,j)]
            m = message(mp, get_feature(X, i), get_feature(X, j), get_feature(E, k))
            push!(E_, m)
        end
    end
    hcat(E_...)
end

@inline function update_batch_vertex(mp::T, M::AbstractMatrix, X::AbstractMatrix) where {T<:MessagePassing}
    X_ = Vector[]
    for i = 1:size(X,2)
        x = update(mp, get_feature(M, i), get_feature(X, i))
        push!(X_, x)
    end
    hcat(X_...)
end

@inline function aggregate_neighbors(mp::T, aggr::Symbol, M::AbstractMatrix, accu_edge, num_V, num_E) where {T<:MessagePassing}
    @assert !iszero(accu_edge) "accumulated edge must not be zero."
    cluster = generate_cluster(M, accu_edge, num_V, num_E)
    pool(aggr, cluster, M)
end

function propagate(mp::T, fg::FeaturedGraph, aggr::Symbol=:add) where {T<:MessagePassing}
    adj = adjacency_list(fg)
    num_V = nv(fg)
    accu_edge = accumulated_edges(adj)
    num_E = accu_edge[end]
    E = edge_feature(fg)
    X = node_feature(fg)

    E = update_batch_edge(mp, adj, E, X)

    M = aggregate_neighbors(mp, aggr, E, accu_edge, num_V, num_E)

    X = update_batch_vertex(mp, M, X)

    FeaturedGraph(graph(fg), X, E, zeros(0))
end
