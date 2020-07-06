abstract type MessagePassing <: GraphNet end

"""
    message(m, x_i, x_j, e_ij)

Message function for message-passing scheme. This function can be overrided to dispatch to custom layers.
First argument should be message-passing layer, the rest of arguments can be `x_i`, `x_j` and `e_ij`.

# Arguments
- `m`: message-passing layer.
- `x_i`: the feature of node `x_i`.
- `x_j`: the feature of neighbors of node `x_i`.
- `e_ij`: the feature of edge (`x_i`, `x_j`).
"""
@inline message(m::T, x_i, x_j, e_ij) where {T<:MessagePassing} = x_j

"""
    update(m, X, M)

Update function for message-passing scheme. This function can be overrided to dispatch to custom layers.
First argument should be message-passing layer, the rest of arguments can be `X` and `M`.

# Arguments
- `m`: message-passing layer.
- `x`: the single node feature.
- `msg`: the message aggregated from message function.
"""
@inline update(m::T, x, msg) where {T<:MessagePassing} = msg

@inline function update_batch_edge(m::T, E::AbstractMatrix, X::AbstractMatrix, adj) where {T<:MessagePassing}
    edge_idx = edge_index_table(adj)
    M = Vector[]
    for (i, js) = enumerate(adj)
        for j = js
            k = edge_idx[(i,j)]
            m = message(m, get_feature(X, i), get_feature(X, j), get_feature(E, k))
            push!(M, m)
        end
    end
    hcat(M...)
end

@inline function update_batch_vertex(m::T, M::AbstractMatrix, X::AbstractMatrix) where {T<:MessagePassing}
    X_ = Vector[]
    for i = 1:size(X,2)
        x = update(m, get_feature(M, i), get_feature(X, i))
        push!(X_, x)
    end
    hcat(X_...)
end

@inline function aggregate_neighbors(m::T, aggr::Symbol, M::AbstractMatrix, accu_edge, num_V, num_E) where {T<:MessagePassing}
    @assert !iszero(accu_edge) "accumulated edge must not be zero."
    cluster = generate_cluster(M, accu_edge, num_V, num_E)
    pool(aggr, cluster, M)
end

function propagate(mp::T, fg::FeaturedGraph, aggr::Symbol=:add) where {T<:MessagePassing}
    _propagate(mp, fg, aggr, nothing, nothing)
end
