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

function sumpool(cluster::AbstractArray{Int}, X::AbstractArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    Y = zeros(T, dims.us_dims[1], c)
    scatter_add!(Y, X, cluster)
    Y
end

function subpool(cluster::AbstractArray{Int}, X::AbstractArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    Y = zeros(T, dims.us_dims[1], c)
    scatter_sub!(Y, X, cluster)
    Y
end

function prodpool(cluster::AbstractArray{Int}, X::AbstractArray{T},
                  c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    Y = ones(T, dims.us_dims[1], c)
    scatter_mul!(Y, X, cluster)
    Y
end

function divpool(cluster::AbstractArray{Int}, X::AbstractArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    FT = floattype(T)
    Y = ones(FT, dims.us_dims[1], c)
    scatter_div!(Y, FT.(X), cluster)
    Y
end

function maxpool(cluster::AbstractArray{Int}, X::AbstractArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    Y = fill(typemin(T), dims.us_dims[1], c)
    scatter_max!(Y, X, cluster)
    Y
end

function minpool(cluster::AbstractArray{Int}, X::AbstractArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    Y = fill(typemax(T), dims.us_dims[1], c)
    scatter_min!(Y, X, cluster)
    Y
end

function meanpool(cluster::AbstractArray{Int}, X::AbstractArray{T},
                  c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    FT = floattype(T)
    Y = zeros(FT, dims.us_dims[1], c)
    scatter_mean!(Y, FT.(X), cluster)
    Y
end

struct Dims
    xs_dims::Tuple
    us_dims::Tuple

    function Dims(xs_dims::Tuple, us_dims::Tuple)
        @assert xs_dims == us_dims[2:end] "X must have the same latter dimension with cluster."
        new(xs_dims, us_dims)
    end
end

Dims(xs::AbstractArray{Int}, us::AbstractArray) = Dims(size(xs), size(us))

@adjoint sumpool(cluster::AbstractArray{Int}, X::AbstractArray{T}) where {T<:Real} =
    sumpool(cluster, X), Δ -> (nothing, gather(zero(Δ)+Δ, cluster))
@adjoint subpool(cluster::AbstractArray{Int}, X::AbstractArray{T}) where {T<:Real} =
    subpool(cluster, X), Δ -> (nothing, -gather(zero(Δ)+Δ, cluster))

@adjoint function prodpool(cluster::Array{Int}, X::Array{T}) where {T<:Real}
    prodpool(cluster, X), function (Δ)
        rev_cluster = gather_indices(cluster)
        ∇X = gather(zero(Δ)+Δ, cluster)
        @inbounds for ind = CartesianIndices(cluster)
            inds = filter(x -> x != ind, rev_cluster[cluster[ind]])
            for i = 1:size(X, 1)
                ∇X[i, ind] *= prod(j -> X[i, j], inds)
            end
        end
        (nothing, ∇X)
    end
end

@adjoint function divpool(cluster::Array{Int}, X::Array{T}) where {T<:Real}
    divpool(cluster, X), function (Δ)
        rev_cluster = gather_indices(cluster)
        ∇X = -gather(zero(Δ)+Δ, cluster) ./ X.^2
        @inbounds for ind = CartesianIndices(cluster)
            inds = filter(x -> x != ind, rev_cluster[cluster[ind]])
            for i = 1:size(X, 1)
                ∇X[i, ind] /= prod(j -> X[i, j], inds)
            end
        end
        (nothing, ∇X)
    end
end

@adjoint function maxpool(cluster::Array{Int}, X::Array{T}) where {T<:Real}
    max = maxpool(cluster, X)
    max, function (Δ)
       Δu = (X .== gather(max, cluster)) .* gather(zero(Δ)+Δ, cluster)
       (nothing, Δu)
    end
end

@adjoint function minpool(cluster::Array{Int}, X::Array{T}) where {T<:Real}
    min = minpool(cluster, X)
    min, function (Δ)
       Δu = (X .== gather(min, cluster)) .* gather(zero(Δ)+Δ, cluster)
       (nothing, Δu)
    end
end

@adjoint function meanpool(cluster::AbstractArray{Int}, X::AbstractArray{T}) where {T<:Real}
    m = meanpool(cluster, X)
    m, function (Δ)
        ΔX = gather(zero(Δ)+Δ, cluster)
        counts = zero.(cluster)
        @inbounds for i = 1:size(m, 2)
            counts += sum(cluster.==i) * (cluster.==i)
        end
        @inbounds for ind = CartesianIndices(counts)
            ΔX[:, ind] ./= counts[ind]
        end
        (nothing, ΔX)
    end
end

function pool(op::Symbol, cluster::AbstractArray, X::AbstractArray)
    if op == :add
        return sumpool(cluster, X)
    elseif op == :sub
        return subpool(cluster, X)
    elseif op == :mul
        return prodpool(cluster, X)
    elseif op == :div
        return divpool(cluster, X)
    elseif op == :max
        return maxpool(cluster, X)
    elseif op == :min
        return minpool(cluster, X)
    elseif op == :mean
        return meanpool(cluster, X)
    end
end
