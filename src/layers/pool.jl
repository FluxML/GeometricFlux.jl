samesize_float = Dict(Int8=>Float16, UInt8=>Float16, Int16=>Float16, UInt16=>Float16,
                      Int32=>Float32, UInt32=>Float32, Int64=>Float64, UInt64=>Float64)

# GlobalPool(x, aggr, batch, size=nothing) # aggr=sum, mean, max
#
# TopKPool()



function sumpool(cluster::Array{Int}, X::Array{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    Y = zeros(T, dims..., c)
    scatter_add!(Y, X, cluster)
    Y
end

function subpool(cluster::Array{Int}, X::Array{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    Y = zeros(T, dims..., c)
    scatter_sub!(Y, X, cluster)
    Y
end

function prodpool(cluster::Array{Int}, X::Array{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    Y = ones(T, dims..., c)
    scatter_mul!(Y, X, cluster)
    Y
end

function divpool(cluster::Array{Int}, X::Array{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    Y = ones(T, dims..., c)
    scatter_div!(Y, X, cluster)
    Y
end

function maxpool(cluster::Array{Int}, X::Array{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    Y = fill(T(-Inf), dims..., c)
    scatter_max!(Y, X, cluster)
    Y
end

function minpool(cluster::Array{Int}, X::Array{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    Y = fill(T(-Inf), dims..., c)
    scatter_min!(Y, X, cluster)
    Y
end

function meanpool(cluster::Array{Int}, X::Array{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    FT = (T <: Integer) ? samesize_float[T] : T
    Y = zeros(FT, dims..., c)
    scatter_mean!(Y, FT.(X), cluster)
    Y
end

function _pooling_dim_check(cluster::Array{Int}, X::Array{T}) where {T<:Real}
    dim_c = size(cluster)
    d = length(dim_c)
    dim_X = size(X)
    n = length(dim_X) - d
    @assert n > 0 "X must have more dimensions than cluster."
    @assert dim_c == dim_X[n+1:end] "X must have the same latter dimension with cluster."
    dim_X[1:n]
end
