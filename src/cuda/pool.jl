function sumpool(cluster::CuArray{Int}, X::CuArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = pooling_dim_check(cluster, X)
    Y = CuArrays.zeros(T, dims.us_dims[1], c)
    scatter_add!(Y, X, cluster)
    Y
end

sumpool(cluster::Array{Int}, X::CuArray{T}, c::Integer=length(Set(cluster))) where {T<:Real} =
    sumpool(CuArray{Int64}(cluster), X, c)

function subpool(cluster::CuArray{Int}, X::CuArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = pooling_dim_check(cluster, X)
    Y = CuArrays.zeros(T, dims.us_dims[1], c)
    scatter_sub!(Y, X, cluster)
    Y
end

subpool(cluster::Array{Int}, X::CuArray{T}, c::Integer=length(Set(cluster))) where {T<:Real} =
    subpool(CuArray{Int64}(cluster), X, c)

function prodpool(cluster::CuArray{Int}, X::CuArray{T},
                  c::Integer=length(Set(cluster))) where {T<:Real}
    dims = pooling_dim_check(cluster, X)
    Y = CuArrays.ones(T, dims.us_dims[1], c)
    scatter_mul!(Y, X, cluster)
    Y
end

prodpool(cluster::Array{Int}, X::CuArray{T}, c::Integer=length(Set(cluster))) where {T<:Real} =
    prodpool(CuArray{Int64}(cluster), X, c)

function divpool(cluster::CuArray{Int}, X::CuArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = pooling_dim_check(cluster, X)
    FT = (T <: Integer) ? INT2FLOAT[T] : T
    Y = CuArrays.ones(FT, dims.us_dims[1], c)
    scatter_div!(Y, FT.(X), cluster)
    Y
end

divpool(cluster::Array{Int}, X::CuArray{T}, c::Integer=length(Set(cluster))) where {T<:Real} =
    divpool(CuArray{Int64}(cluster), X, c)

function maxpool(cluster::CuArray{Int}, X::CuArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = pooling_dim_check(cluster, X)
    Y = CuArrays.fill(typemin(T), dims.us_dims[1], c)
    scatter_max!(Y, X, cluster)
    Y
end

maxpool(cluster::Array{Int}, X::CuArray{T}, c::Integer=length(Set(cluster))) where {T<:Real} =
    maxpool(CuArray{Int64}(cluster), X, c)

function minpool(cluster::CuArray{Int}, X::CuArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = pooling_dim_check(cluster, X)
    Y = CuArrays.fill(typemax(T), dims.us_dims[1], c)
    scatter_min!(Y, X, cluster)
    Y
end

minpool(cluster::Array{Int}, X::CuArray{T}, c::Integer=length(Set(cluster))) where {T<:Real} =
    minpool(CuArray{Int64}(cluster), X, c)

function meanpool(cluster::CuArray{Int}, X::CuArray{T},
                  c::Integer=length(Set(cluster))) where {T<:Real}
    dims = pooling_dim_check(cluster, X)
    FT = (T <: Integer) ? INT2FLOAT[T] : T
    Y = CuArrays.zeros(FT, dims.us_dims[1], c)
    scatter_mean!(Y, FT.(X), cluster)
    Y
end

meanpool(cluster::Array{Int}, X::CuArray{T}, c::Integer=length(Set(cluster))) where {T<:Real} =
    meanpool(CuArray{Int64}(cluster), X, c)

@adjoint function prodpool(cluster::CuArray{Int}, X::CuArray{T}) where {T<:Real}
    prodpool(cluster, X), function (Δ)
        rev_cluster = gather_indices(cluster)
        ∇X = gather(zero(Δ)+Δ, cluster)
        @inbounds for ind = CartesianIndices(cluster)
            ind = Tuple(ind)
            inds = filter(x -> x != ind, rev_cluster[cluster[ind...]])
            for i = 1:size(X, 1)
                ∇X[i, ind...] *= mapreduce(j -> X[i, j...], *, inds; init=one(T))
            end
        end
        (nothing, ∇X)
    end
end

@adjoint function divpool(cluster::CuArray{Int}, X::CuArray{T}) where {T<:Real}
    divpool(cluster, X), function (Δ)
        rev_cluster = gather_indices(cluster)
        ∇X = -gather(zero(Δ)+Δ, cluster)
        ∇X ./= X.^2
        @inbounds for ind = CartesianIndices(cluster)
            ind = Tuple(ind)
            inds = filter(x -> x != ind, rev_cluster[cluster[ind...]])
            for i = 1:size(X, 1)
                ∇X[i, ind...] /= mapreduce(j -> X[i, j...], *, inds; init=one(T))
            end
        end
        (nothing, ∇X)
    end
end

@adjoint function maxpool(cluster::CuArray{Int}, X::CuArray{T}) where {T<:Real}
    max = maxpool(cluster, X)
    max, function (Δ)
       Δu = (X .== gather(max, cluster))
       Δu .*= gather(zero(Δ)+Δ, cluster)
       (nothing, Δu)
    end
end

@adjoint function minpool(cluster::CuArray{Int}, X::CuArray{T}) where {T<:Real}
    min = minpool(cluster, X)
    min, function (Δ)
       Δu = (X .== gather(min, cluster))
       Δu .*= gather(zero(Δ)+Δ, cluster)
       (nothing, Δu)
    end
end
