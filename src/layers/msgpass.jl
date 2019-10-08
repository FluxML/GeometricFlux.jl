abstract type MessagePassing <: Meta end

adjlist(m::T) where {T<:MessagePassing} = m.adjlist
message(m::T; kwargs...) where {T<:MessagePassing} = identity(; kwargs...)
update(m::T; kwargs...) where {T<:MessagePassing} = identity(; kwargs...)

update_edge(m::T; kwargs...) where {T<:MessagePassing} = message(m; kwargs...)
update_vertex(m::T; kwargs...) where {T<:MessagePassing} = update(m; kwargs...)
update_global(m::T; kwargs...) where {T<:MessagePassing} = identity(; kwargs...)

aggregate_e2v(m::T, aggr::Symbol; kwargs...) where {T<:MessagePassing} =
    neighboring(m, aggr; kwargs...)

function propagate(mp::T; aggr::Symbol=:add, kwargs...) where {T<:MessagePassing}
    adj = adjlist(mp)
    msg_args = getdata(kwargs, 1, adj[1])

    M = message(mp; msg_args...)
    M, cluster = apply_messages(mp, M, adj; kwargs...)

    M = neighboring(mp, aggr; M=M, cluster=cluster)

    upd_args = haskey(kwargs, :X) ? (M=M, X=kwargs[:X]) : (M=M, )
    Y = update(mp; upd_args...)
    return Y
end

neighboring(mp::T, aggr::Symbol; kwargs...) where {T<:MessagePassing} =
    pool(aggr, kwargs[:cluster], kwargs[:M])

function getdata(d, i::Integer, ne)
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

function apply_messages(mp, M::AbstractMatrix{T}, adj::AbstractVector{<:AbstractVector},
                        F::Integer=size(M,1), N::Integer=length(adj), n::Integer=length(adj[1]);
                        kwargs...) where {T<:Real}
    if F > 50
        # TODO: Benchmark how many N or F need to switch from sequential to milti-thread version.
        return thread_apply_messages(mp, M, adj, F, N, n; kwargs...)
    else
        ne_N = mapreduce(length, +, adj)
        Y = M
        cluster = Vector{Int}(undef, ne_N)
        cluster[1:n] .= 1
        j = n
        @inbounds for i = 2:N
            ne = adj[i]
            n = length(ne)
            msg_args = getdata(kwargs, i, ne)
            Y = hcat(Y, message(mp; msg_args...))
            cluster[j+1:j+n] .= i
            j += n
        end
        return Y, cluster
    end
end

function thread_apply_messages(mp, M::AbstractMatrix{T}, adj::AbstractVector{<:AbstractVector},
                               F::Integer=size(M,1), N::Integer=length(adj), n::Integer=length(adj[1]);
                               kwargs...) where {T<:Real}
    starts = cumsum(map(length, adj))
    ne_N = starts[end]
    Y = Matrix{T}(undef, F, ne_N)
    Y[:, 1:n] = M
    cluster = Vector{Int}(undef, ne_N)
    cluster[1:n] .= 1
    @inbounds Threads.@threads for i = 2:N
        j = starts[i-1]
        k = starts[i]
        msg_args = getdata(kwargs, i, adj[i])
        Y[:, j+1:k] = message(mp; msg_args...)
        cluster[j+1:k] .= i
    end
    Y, cluster
end
