abstract type MessagePassing end

adjlist(m::T) where {T<:MessagePassing} = m.adjlist
message(m::T) where {T<:MessagePassing} = error("not implement")
update(m::T) where {T<:MessagePassing} = error("not implement")

function propagate(mp::T; aggr::Symbol=:add, kwargs...) where {T<:MessagePassing}
    M, cluster = neighboring(mp; kwargs...)
    M = pool(aggr, cluster, M)
    upd_args = Dict{Symbol,AbstractArray}(:M=>M)
    haskey(kwargs, :X) && (upd_args[:X] = kwargs[:X])
    Y = update(mp; upd_args...)
    return Y
end

function neighboring(mp::T; kwargs...) where {T<:MessagePassing}
    adj = adjlist(mp)
    msg_args = getdata(kwargs, 1, adj[1])
    M = message(mp; msg_args...)
    apply_messages(mp, M, adj; kwargs...)
end

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
        Y = Matrix{T}(undef, F, ne_N)
        Y[:, 1:n] = M
        cluster = Vector{Int}(undef, ne_N)
        cluster[1:n] .= 1
        j = n
        @inbounds for i = 2:N
            ne = adj[i]
            n = length(ne)
            msg_args = getdata(kwargs, i, ne)
            Y[:, j+1:j+n] = message(mp; msg_args...)
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
