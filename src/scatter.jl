function scatter(op::typeof(+), cluster::AbstractArray{Int}, X::AbstractArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    Y = zeros(T, dims.us_dims[1], c)
    ScatterNNlib.scatter_add!(Y, X, cluster)
    Y
end

function scatter(op::typeof(-), cluster::AbstractArray{Int}, X::AbstractArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    Y = zeros(T, dims.us_dims[1], c)
    ScatterNNlib.scatter_sub!(Y, X, cluster)
    Y
end

function scatter(op::typeof(*), cluster::AbstractArray{Int}, X::AbstractArray{T},
                  c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    Y = ones(T, dims.us_dims[1], c)
    ScatterNNlib.scatter_mul!(Y, X, cluster)
    Y
end

function scatter(op::typeof(/), cluster::AbstractArray{Int}, X::AbstractArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    FT = float(T)
    Y = ones(FT, dims.us_dims[1], c)
    ScatterNNlib.scatter_div!(Y, FT.(X), cluster)
    Y
end

function scatter(op::typeof(max), cluster::AbstractArray{Int}, X::AbstractArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    Y = fill(typemin(T), dims.us_dims[1], c)
    ScatterNNlib.scatter_max!(Y, X, cluster)
    Y
end

function scatter(op::typeof(min), cluster::AbstractArray{Int}, X::AbstractArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    Y = fill(typemax(T), dims.us_dims[1], c)
    ScatterNNlib.scatter_min!(Y, X, cluster)
    Y
end

function scatter(op::typeof(mean), cluster::AbstractArray{Int}, X::AbstractArray{T},
                 c::Integer=length(Set(cluster))) where {T<:Real}
    dims = Dims(cluster, X)
    FT = float(T)
    Y = zeros(FT, dims.us_dims[1], c)
    ScatterNNlib.scatter_mean!(Y, FT.(X), cluster)
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

@adjoint scatter(op::typeof(+), cluster::AbstractArray{Int}, X::AbstractArray{T}) where {T<:Real} =
    scatter(op, cluster, X), Δ -> (nothing, nothing, ScatterNNlib.gather(zero(Δ)+Δ, cluster))
@adjoint scatter(op::typeof(-), cluster::AbstractArray{Int}, X::AbstractArray{T}) where {T<:Real} =
    scatter(op, cluster, X), Δ -> (nothing, nothing, -ScatterNNlib.gather(zero(Δ)+Δ, cluster))

@adjoint function scatter(op::typeof(*), cluster::Array{Int}, X::Array{T}) where {T<:Real}
    scatter(op, cluster, X), function (Δ)
        rev_cluster = ScatterNNlib.gather_indices(cluster)
        ∇X = ScatterNNlib.gather(zero(Δ)+Δ, cluster)
        @inbounds for ind = CartesianIndices(cluster)
            inds = filter(x -> x != ind, rev_cluster[cluster[ind]])
            for i = 1:size(X, 1)
                ∇X[i, ind] *= prod(j -> X[i, j], inds)
            end
        end
        (nothing, nothing, ∇X)
    end
end

@adjoint function scatter(op::typeof(/), cluster::Array{Int}, X::Array{T}) where {T<:Real}
    scatter(op, cluster, X), function (Δ)
        rev_cluster = ScatterNNlib.gather_indices(cluster)
        ∇X = -ScatterNNlib.gather(zero(Δ)+Δ, cluster) ./ X.^2
        @inbounds for ind = CartesianIndices(cluster)
            inds = filter(x -> x != ind, rev_cluster[cluster[ind]])
            for i = 1:size(X, 1)
                ∇X[i, ind] /= prod(j -> X[i, j], inds)
            end
        end
        (nothing, nothing, ∇X)
    end
end

@adjoint function scatter(op::typeof(max), cluster::Array{Int}, X::Array{T}) where {T<:Real}
    max = scatter(op, cluster, X)
    max, function (Δ)
       Δu = (X .== ScatterNNlib.gather(max, cluster)) .* ScatterNNlib.gather(zero(Δ)+Δ, cluster)
       (nothing, nothing, Δu)
    end
end

@adjoint function scatter(op::typeof(min), cluster::Array{Int}, X::Array{T}) where {T<:Real}
    min = scatter(op, cluster, X)
    min, function (Δ)
       Δu = (X .== ScatterNNlib.gather(min, cluster)) .* ScatterNNlib.gather(zero(Δ)+Δ, cluster)
       (nothing, nothing, Δu)
    end
end

@adjoint function scatter(op::typeof(mean), cluster::AbstractArray{Int}, X::AbstractArray{T}) where {T<:Real}
    m = scatter(op, cluster, X)
    m, function (Δ)
        ΔX = ScatterNNlib.gather(zero(Δ)+Δ, cluster)
        counts = zero.(cluster)
        @inbounds for i = 1:size(m, 2)
            counts += sum(cluster.==i) * (cluster.==i)
        end
        @inbounds for ind = CartesianIndices(counts)
            ΔX[:, ind] ./= counts[ind]
        end
        (nothing, nothing, ΔX)
    end
end
