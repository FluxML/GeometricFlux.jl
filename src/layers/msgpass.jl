abstract type MessagePassing <: Meta end

adjlist(m::T) where {T<:MessagePassing} = m.adjlist
message(m::T; kwargs...) where {T<:MessagePassing} = identity(; kwargs...)
update(m::T; kwargs...) where {T<:MessagePassing} = identity(; kwargs...)

update_edge(m::T; kwargs...) where {T<:MessagePassing} = message(m; kwargs...)
update_vertex(m::T; kwargs...) where {T<:MessagePassing} = update(m; kwargs...)
update_global(m::T; kwargs...) where {T<:MessagePassing} = identity(; kwargs...)

aggregate_neighbors(m::T, aggr::Symbol; kwargs...) where {T<:MessagePassing} =
    pool(aggr, kwargs[:cluster], kwargs[:M])


struct GraphInfo
    adj::AbstractVector{<:AbstractVector}
    edge_idx::AbstractVector{<:Integer}
    V::Integer
    E::Integer

    function GraphInfo(adj)
        edge_idx = edge_index_table(adj)
        V = length(adj)
        E = edge_idx[end]
        new(adj, edge_idx, V, E)
    end
end

edge_index_table(adj) = append!([0], cumsum(map(length, adj)))


function propagate(mp::T; aggr::Symbol=:add, kwargs...) where {T<:MessagePassing}
    gi = GraphInfo(adjlist(mp))

    # message function
    M = message(mp; neighbor_data(kwargs, 1, gi.adj[1])...)
    M = apply_messages(mp, M, gi; kwargs...)

    # aggregate function
    M = aggregate_neighbors(mp, aggr; M=M, cluster=cluster_table(M, gi))

    # update function
    Y = update_vertex(mp; M=M, kwargs...)
    return Y
end

function neighbor_data(d, i::Integer, ne)
    result = Dict{Symbol,AbstractArray}()
    if haskey(d, :X)
        result[:x_i] = view(d[:X], :, i)
        result[:x_j] = view(d[:X], :, ne)
    end
    if haskey(d, :E)
        result[:e_ij] = view(d[:E], :, i, ne)
    end
    result
end

function cluster_table(M::AbstractMatrix, gi::GraphInfo)
    cluster = similar(M, Int, gi.E)
    @inbounds for i = 1:gi.V
        j = gi.edge_idx[i]
        k = gi.edge_idx[i+1]
        cluster[j+1:k] .= i
    end
    cluster
end

function apply_messages(mp, M::AbstractMatrix, gi::GraphInfo, F::Integer=size(M,1);
                        kwargs...)
    adj = gi.adj
    Y = similar(M, F, gi.E)
    @inbounds Threads.@threads for i = 1:gi.V
        j = gi.edge_idx[i]
        k = gi.edge_idx[i+1]
        msg_args = neighbor_data(kwargs, i, adj[i])
        Y[:, j+1:k] = message(mp; msg_args...)
    end
    Y
end
