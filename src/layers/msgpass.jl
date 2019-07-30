const aggr_func = Dict{Symbol,Function}(:+ => sum, :max => maximum, :mean => mean)

abstract type MessagePassing end

adjlist(m::T) where {T<:MessagePassing} = m.adjlist
message(m::T) where {T<:MessagePassing} = error("not implement")
update(m::T) where {T<:MessagePassing} = error("not implement")

function propagate(mp::T; aggr::Symbol=:add, kwargs...) where {T<:MessagePassing}
    M, cluster = neighboring(mp; kwargs...)
    M = pool(aggr, cluster, M)
    upd_args = Dict{Symbol,AbstractArray}(:M=>M')
    haskey(kwargs, :X) && (upd_args[:X] = kwargs[:X])
    Y = update(mp; upd_args...)
    return Y
end

function neighboring(mp::T; kwargs...) where {T<:MessagePassing}  # TODO: need refactor
    adj = adjlist(mp)
    N = length(adj)
    ne_N = mapreduce(length, +, adj)
    ne = adj[1]
    n = length(ne)
    msg_args = getdata(kwargs, 1, ne)
    M = message(mp; msg_args...)
    F = size(M,1)
    Y = Matrix{eltype(M)}(undef, F, ne_N)
    Y[:, 1:n] = M
    cluster = Vector{Int}(undef, ne_N)
    cluster[1:n] .= 1
    j = n
    # TODO: Benchmark how many N or F need to switch from sequential to milti-thread version.
    @inbounds for i = 2:N
        ne = adj[i]
        n = length(ne)
        msg_args = getdata(kwargs, i, ne)
        Y[:, j+1:j+n] = message(mp; msg_args...)
        cluster[j+1:j+n] .= i
        j += n
    end
    Y, cluster
end

function getdata(d, i::Integer, ne)
    result = Dict{Symbol,AbstractArray}()
    if haskey(d, :X)
        result[:x_i] = view(d[:X]', :, i)
        result[:x_j] = view(d[:X]', :, ne)
    end
    if haskey(d, :E)
        result[:e_ij] = view(d[:E], :, i, ne)
    end
    result
end
