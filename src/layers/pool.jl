using DataStructures: nlargest

struct GlobalPool{A}
    aggr
    cluster::A
    function GlobalPool(aggr, dim...)
        cluster = ones(Int64, dim)
        new{typeof(cluster)}(aggr, cluster)
    end
end

(g::GlobalPool)(X::AbstractArray) = NNlib.scatter(g.aggr, X, g.cluster)

struct LocalPool{A}
    aggr
    cluster::A
end

function LocalPool(aggr, cluster::AbstractArray)
    LocalPool{typeof(cluster)}(aggr, cluster)
end

(l::LocalPool)(X::AbstractArray) = NNlib.scatter(l.aggr, X, l.cluster)

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

function topk_index(y::AbstractVector, k::Integer)
    v = nlargest(k, y)
    return collect(1:length(y))[y .>= v[end]]
end

topk_index(y::Adjoint, k::Integer) = topk_index(y', k)
