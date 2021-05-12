function scatter(op::typeof(+), cluster::CuArray{Int}, X::CuArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    Y = CUDA.zeros(T, dims.us_dims[1], c)
    ScatterNNlib.scatter_add!(Y, X, cluster)
    Y
end

scatter(op::typeof(+), cluster::Array{Int}, X::CuArray{T}, c::Integer=length(Set(cluster))) where {T<:Real} =
    scatter(op, CuArray{Int64}(cluster), X, c)

function scatter(op::typeof(-), cluster::CuArray{Int}, X::CuArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    Y = CUDA.zeros(T, dims.us_dims[1], c)
    ScatterNNlib.scatter_sub!(Y, X, cluster)
    Y
end

scatter(op::typeof(-), cluster::Array{Int}, X::CuArray{T}, c::Integer=length(Set(cluster))) where {T<:Real} =
    scatter(op, CuArray{Int64}(cluster), X, c)

function scatter(op::typeof(*), cluster::CuArray{Int}, X::CuArray{T},
                  c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    Y = CUDA.ones(T, dims.us_dims[1], c)
    ScatterNNlib.scatter_mul!(Y, X, cluster)
    Y
end

scatter(op::typeof(*), cluster::Array{Int}, X::CuArray{T}, c::Integer=length(Set(cluster))) where {T<:Real} =
    scatter(op, CuArray{Int64}(cluster), X, c)

function scatter(op::typeof(/), cluster::CuArray{Int}, X::CuArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    FT = float(T)
    Y = CUDA.ones(FT, dims.us_dims[1], c)
    ScatterNNlib.scatter_div!(Y, FT.(X), cluster)
    Y
end

scatter(op::typeof(/), cluster::Array{Int}, X::CuArray{T}, c::Integer=length(Set(cluster))) where {T<:Real} =
    scatter(op, CuArray{Int64}(cluster), X, c)

function scatter(op::typeof(max), cluster::CuArray{Int}, X::CuArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    Y = CUDA.fill(typemin(T), dims.us_dims[1], c)
    ScatterNNlib.scatter_max!(Y, X, cluster)
    Y
end

scatter(op::typeof(max), cluster::Array{Int}, X::CuArray{T}, c::Integer=length(Set(cluster))) where {T<:Real} =
    scatter(op, CuArray{Int64}(cluster), X, c)

function scatter(op::typeof(min), cluster::CuArray{Int}, X::CuArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    Y = CUDA.fill(typemax(T), dims.us_dims[1], c)
    ScatterNNlib.scatter_min!(Y, X, cluster)
    Y
end

scatter(op::typeof(min), cluster::Array{Int}, X::CuArray{T}, c::Integer=length(Set(cluster))) where {T<:Real} =
    scatter(op, CuArray{Int64}(cluster), X, c)

function scatter(op::typeof(mean), cluster::CuArray{Int}, X::CuArray{T},
                  c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    FT = float(T)
    Y = CUDA.zeros(FT, dims.us_dims[1], c)
    ScatterNNlib.scatter_mean!(Y, FT.(X), cluster)
    Y
end

scatter(op::typeof(mean), cluster::Array{Int}, X::CuArray{T}, c::Integer=length(Set(cluster))) where {T<:Real} =
    scatter(op::typeof(mean), CuArray{Int64}(cluster), X, c)

@adjoint function scatter(op::typeof(*), cluster::CuArray{Int}, X::CuArray{T}) where {T<:Real}
    scatter(op, cluster, X), function (Δ)
        rev_cluster = ScatterNNlib.gather_indices(cluster)
        ∇X = ScatterNNlib.gather(zero(Δ)+Δ, cluster)
        @inbounds for ind = CartesianIndices(cluster)
            ind = Tuple(ind)
            inds = [x for x in rev_cluster[cluster[ind...]] if x != ind]
            for i = 1:size(X, 1)
                multiplier = one(T)
                for j = inds
                    multiplier *= X[i, j...]
                end
                ∇X[i, ind...] *= multiplier
            end
        end
        (nothing, nothing, ∇X)
    end
end

@adjoint function scatter(op::typeof(/), cluster::CuArray{Int}, X::CuArray{T}) where {T<:Real}
    scatter(op, cluster, X), function (Δ)
        rev_cluster = ScatterNNlib.gather_indices(cluster)
        ∇X = -ScatterNNlib.gather(zero(Δ)+Δ, cluster)
        ∇X ./= X.^2
        @inbounds for ind = CartesianIndices(cluster)
            ind = Tuple(ind)
            inds = [x for x in rev_cluster[cluster[ind...]] if x != ind]
            for i = 1:size(X, 1)
                denom = one(T)
                for j = inds
                    denom *= X[i, j...]
                end
                ∇X[i, ind...] /= denom
            end
        end
        (nothing, nothing, ∇X)
    end
end

@adjoint function scatter(op::typeof(max), cluster::CuArray{Int}, X::CuArray{T}) where {T<:Real}
    max = scatter(op, cluster, X)
    max, function (Δ)
       Δu = (X .== ScatterNNlib.gather(max, cluster))
       Δu .*= ScatterNNlib.gather(zero(Δ)+Δ, cluster)
       (nothing, nothing, Δu)
    end
end

@adjoint function scatter(op::typeof(min), cluster::CuArray{Int}, X::CuArray{T}) where {T<:Real}
    min = scatter(op, cluster, X)
    min, function (Δ)
       Δu = (X .== ScatterNNlib.gather(min, cluster))
       Δu .*= ScatterNNlib.gather(zero(Δ)+Δ, cluster)
       (nothing, nothing, Δu)
    end
end
