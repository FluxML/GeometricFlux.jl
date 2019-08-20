samesize_float = Dict(Int8=>Float16, UInt8=>Float16, Int16=>Float16, UInt16=>Float16,
                      Int32=>Float32, UInt32=>Float32, Int64=>Float64, UInt64=>Float64)

# GlobalPool(x, aggr, batch, size=nothing) # aggr=sum, mean, max
#
# TopKPool()

# struct MaxPool
#
# end
#
# function MaxPool(adj::AbstractMatrix, ch::Pair{<:Integer,<:Integer}, σ = identity)
#     MaxPool()
# end
#
# (m::MaxPool)(X::AbstractMatrix) = maxpool(m.cluster, X)
#
#
#
# struct MeanPool
#
# end
#
# (m::MeanPool)(X::AbstractMatrix) = meanpool(m.cluster, X)



function sumpool(cluster::AbstractArray{Int}, X::AbstractArray{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    Y = zeros(T, dims..., c)
    scatter_add!(Y, X, cluster)
    Y
end

function subpool(cluster::AbstractArray{Int}, X::AbstractArray{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    Y = zeros(T, dims..., c)
    scatter_sub!(Y, X, cluster)
    Y
end

function prodpool(cluster::AbstractArray{Int}, X::AbstractArray{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    Y = ones(T, dims..., c)
    scatter_mul!(Y, X, cluster)
    Y
end

function divpool(cluster::AbstractArray{Int}, X::AbstractArray{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    FT = (T <: Integer) ? samesize_float[T] : T
    Y = ones(FT, dims..., c)
    scatter_div!(Y, FT.(X), cluster)
    Y
end

function maxpool(cluster::AbstractArray{Int}, X::AbstractArray{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    Y = fill(typemin(T), dims..., c)
    scatter_max!(Y, X, cluster)
    Y
end

function minpool(cluster::AbstractArray{Int}, X::AbstractArray{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    Y = fill(typemax(T), dims..., c)
    scatter_min!(Y, X, cluster)
    Y
end

function meanpool(cluster::AbstractArray{Int}, X::AbstractArray{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    FT = (T <: Integer) ? samesize_float[T] : T
    Y = zeros(FT, dims..., c)
    scatter_mean!(Y, FT.(X), cluster)
    Y
end

function _pooling_dim_check(cluster::AbstractArray{Int}, X::AbstractArray{T}) where {T<:Real}
    dim_c = size(cluster)
    d = length(dim_c)
    dim_X = size(X)
    n = length(dim_X) - d
    @assert n > 0 "X must have more dimensions than cluster."
    @assert dim_c == dim_X[n+1:end] "X must have the same latter dimension with cluster."
    dim_X[1:n]
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
