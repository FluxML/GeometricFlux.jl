using Base.Threads

abstract type MessagePassing <: Meta end

adjlist(m::T) where {T<:MessagePassing} = m.adjlist

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
message(m::T, x_i=zeros(0), x_j=zeros(0)) where {T<:MessagePassing} = x_j

"""
    update(m, X, M)

Update function for message-passing scheme. This function can be overrided to dispatch to custom layers.
First argument should be message-passing layer, the rest of arguments can be `X` and `M`.

# Arguments
- `m`: message-passing layer.
- `X`: the feature of all nodes.
- `M`: the message aggregated from message function.
"""
update(m::T, X, M) where {T<:MessagePassing} = M

function update_edge(m::T; gi::GraphInfo, kwargs...) where {T<:MessagePassing}
    adj = gi.adj
    edge_idx = gi.edge_idx
    X = get(kwargs, :X, nothing)
    E = get(kwargs, :E, nothing)
    args = drop_nothing(get_xi(X, 1), get_xj(X, adj[1]), get_eij(E, 1, adj[1]))
    M = message(m, args...)
    dims = collect(size(M))
    dims[end] = gi.E
    Y = similar(M, dims...)
    assign!(Y, M; last_dim=1:edge_idx[2])
    _apply_msg!(m, Y, gi.V, edge_idx, adj, X, E)
end

function _apply_msg!(m, Y::Array, V, edge_idx, adj, X=nothing, E=nothing)
    @inbounds Threads.@threads for i = 2:V
        j = edge_idx[i]
        k = edge_idx[i+1]
        args = drop_nothing(get_xi(X, i), get_xj(X, adj[i]), get_eij(E, i, adj[i]))
        M = message(m, args...)
        assign!(Y, M; last_dim=j+1:k)
    end
    Y
end

function update_vertex(m::T, M::AbstractArray; kwargs...) where {T<:MessagePassing}
    X = get(kwargs, :X, nothing)
    update(m, X, M)
end

aggregate_neighbors(m::T, aggr::Symbol; kwargs...) where {T<:MessagePassing} =
    pool(aggr, kwargs[:cluster], kwargs[:M])

function propagate(mp::T, aggr::Symbol=:add; adjl=adjlist(mp), kwargs...) where {T<:MessagePassing}
    gi = GraphInfo(adjl)

    # message function
    M = update_edge(mp; gi=gi, kwargs...)

    # aggregate function
    cluster = generate_cluster(M, gi)
    M = aggregate_neighbors(mp, aggr; M=M, cluster=cluster)

    # update function
    Y = update_vertex(mp, M; kwargs...)
    return Y
end
