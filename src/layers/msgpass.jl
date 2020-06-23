using Base.Threads

abstract type MessagePassing <: Meta end

adjlist(m::T) where {T<:MessagePassing} = m.adjlist
message(m::T; kwargs...) where {T<:MessagePassing} = identity(; kwargs...)
update(m::T; kwargs...) where {T<:MessagePassing} = identity(; kwargs...)

function update_edge(m::T; gi::GraphInfo, kwargs...) where {T<:MessagePassing}
    adj = gi.adj
    edge_idx = gi.edge_idx
    M = message(m; adjacent_vertices_data(kwargs, 1, adj[1])...,
                   incident_edges_data(kwargs, 1, adj[1])...)
    dims = collect(size(M))
    dims[end] = gi.E
    Y = similar(M, dims...)
    assign!(Y, M; last_dim=1:edge_idx[2])
    _apply_msg!(m, Y, gi.V, edge_idx, adj; kwargs...)
end

function _apply_msg!(m, Y::Array, V, edge_idx, adj; kwargs...)
    @inbounds Threads.@threads for i = 2:V
        j = edge_idx[i]
        k = edge_idx[i+1]
        M = message(m; adjacent_vertices_data(kwargs, i, adj[i])...,
                       incident_edges_data(kwargs, i, adj[i])...)
        assign!(Y, M; last_dim=j+1:k)
    end
    Y
end

update_vertex(m::T; kwargs...) where {T<:MessagePassing} = update(m; kwargs...)

aggregate_neighbors(m::T, aggr::Symbol; kwargs...) where {T<:MessagePassing} =
    pool(aggr, kwargs[:cluster], kwargs[:M])

function propagate(mp::T; aggr::Symbol=:add, adjl=adjlist(mp), kwargs...) where {T<:MessagePassing}
    gi = GraphInfo(adjl)

    # message function
    M = update_edge(mp; gi=gi, kwargs...)

    # aggregate function
    cluster = generate_cluster(M, gi)
    M = aggregate_neighbors(mp, aggr; M=M, cluster=cluster)

    # update function
    Y = update_vertex(mp; M=M, kwargs...)
    return Y
end
