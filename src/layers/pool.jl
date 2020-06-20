using DataStructures: nlargest

struct GlobalPool{A}
    aggr::Symbol
    cluster::A
    function GlobalPool(aggr::Symbol, dim...)
        cluster = ones(Int64, dim)
        new{typeof(cluster)}(aggr, cluster)
    end
end

(g::GlobalPool)(X::AbstractArray) = pool(g.aggr, g.cluster, X)

struct LocalPool{A}
    aggr::Symbol
    cluster::A
end

function LocalPool(aggr::Symbol, cluster::AbstractArray)
    LocalPool{typeof(cluster)}(aggr, cluster)
end

(l::LocalPool)(X::AbstractArray) = pool(l.aggr, l.cluster, X)

struct TopKPool{T,S}
    A::AbstractMatrix{T}
    k::Int
    p::AbstractVector{S}
    Ã::AbstractMatrix{T}
end

function TopKPool(adj::AbstractMatrix, k::Int, in_channel::Integer; init=glorot_uniform)
    TopKPool(adj, k, init(in_channel), similar(adj, k, k))
end

function (t::TopKPool)(X::AbstractArray)
    y = t.p' * X / norm(t.p)
    idx = topk_index(y, t.k)
    t.Ã .= view(t.A, idx, idx)
    X_ = view(X, :, idx) .* σ.(view(y, idx)')
    return X_
end
