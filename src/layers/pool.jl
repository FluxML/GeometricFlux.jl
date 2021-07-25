using DataStructures: nlargest

"""
    GlobalPool(aggr, dim...)

Global pooling layer.

It pools all features with `aggr` operation.

# Arguments

- `aggr`: An aggregate function applied to pool all features.
"""
struct GlobalPool{A}
    aggr
    cluster::A
    function GlobalPool(aggr, dim...)
        cluster = ones(Int64, dim)
        new{typeof(cluster)}(aggr, cluster)
    end
end

(g::GlobalPool)(X::AbstractArray) = NNlib.scatter(g.aggr, X, g.cluster)

"""
    LocalPool(aggr, cluster)

Local pooling layer.

It pools features with `aggr` operation accroding to `cluster`. It is implemented with `scatter` operation.

# Arguments

- `aggr`: An aggregate function applied to pool all features.
- `cluster`: An index structure which indicates what features to aggregate with.
"""
struct LocalPool{A<:AbstractArray}
    aggr
    cluster::A
end

function LocalPool(aggr, cluster::AbstractArray)
    LocalPool{typeof(cluster)}(aggr, cluster)
end

(l::LocalPool)(X::AbstractArray) = NNlib.scatter(l.aggr, X, l.cluster)

"""
    TopKPool(adj, k, in_channel)

Top-k pooling layer.

# Arguments

- `adj`: Adjacency matrix  of a graph.
- `k`: Top-k nodes are selected to pool together.
- `in_channel`: The dimension of input channel.
"""
struct TopKPool{T,S}
    A::AbstractMatrix{T}
    k::Int
    p::AbstractVector{S}
    Ã::AbstractMatrix{T}
end

function TopKPool(adj::AbstractMatrix, k::Int, in_channel::Int; init=glorot_uniform)
    TopKPool(adj, k, init(in_channel), similar(adj, k, k))
end

function (t::TopKPool)(X::AbstractArray)
    y = t.p' * X / norm(t.p)
    idx = topk_index(y, t.k)
    t.Ã .= view(t.A, idx, idx)
    X_ = view(X, :, idx) .* σ.(view(y, idx)')
    return X_
end

function topk_index(y::AbstractVector, k::Int)
    v = nlargest(k, y)
    return collect(1:length(y))[y .>= v[end]]
end

topk_index(y::Adjoint, k::Int) = topk_index(y', k)
