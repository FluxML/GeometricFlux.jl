function sumpool(cluster::CuArray{Int}, X::CuArray{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    Y = CuArrays.zeros(T, dims..., c)
    scatter_add!(Y, X, cluster)
    Y
end

sumpool(cluster::Array{Int}, X::CuArray{T}) where {T<:Real} = sumpool(CuArray{Int64}(cluster), X)

function subpool(cluster::CuArray{Int}, X::CuArray{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    Y = CuArrays.zeros(T, dims..., c)
    scatter_sub!(Y, X, cluster)
    Y
end

subpool(cluster::Array{Int}, X::CuArray{T}) where {T<:Real} = subpool(CuArray{Int64}(cluster), X)

function prodpool(cluster::CuArray{Int}, X::CuArray{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    Y = CuArrays.ones(T, dims..., c)
    scatter_mul!(Y, X, cluster)
    Y
end

prodpool(cluster::Array{Int}, X::CuArray{T}) where {T<:Real} = prodpool(CuArray{Int64}(cluster), X)

function divpool(cluster::CuArray{Int}, X::CuArray{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    FT = (T <: Integer) ? INT2FLOAT[T] : T
    Y = CuArrays.ones(FT, dims..., c)
    scatter_div!(Y, FT.(X), cluster)
    Y
end

divpool(cluster::Array{Int}, X::CuArray{T}) where {T<:Real} = divpool(CuArray{Int64}(cluster), X)

function maxpool(cluster::CuArray{Int}, X::CuArray{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    Y = CuArrays.fill(typemin(T), dims..., c)
    scatter_max!(Y, X, cluster)
    Y
end

maxpool(cluster::Array{Int}, X::CuArray{T}) where {T<:Real} = maxpool(CuArray{Int64}(cluster), X)

function minpool(cluster::CuArray{Int}, X::CuArray{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    Y = CuArrays.fill(typemax(T), dims..., c)
    scatter_min!(Y, X, cluster)
    Y
end

minpool(cluster::Array{Int}, X::CuArray{T}) where {T<:Real} = minpool(CuArray{Int64}(cluster), X)

function meanpool(cluster::CuArray{Int}, X::CuArray{T}) where {T<:Real}
    dims = _pooling_dim_check(cluster, X)
    c = length(Set(cluster))
    FT = (T <: Integer) ? INT2FLOAT[T] : T
    Y = CuArrays.zeros(FT, dims..., c)
    scatter_mean!(Y, FT.(X), cluster)
    Y
end

meanpool(cluster::Array{Int}, X::CuArray{T}) where {T<:Real} = meanpool(CuArray{Int64}(cluster), X)
