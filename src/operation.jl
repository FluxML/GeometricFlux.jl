_gather(::Nothing, idx) = nothing
_gather(A::Fill{T,2,Axes}, idx) where {T,Axes} = fill(A.value, A.axes[1], length(idx))
_gather(A::AbstractMatrix, idx) = NNlib.gather(A, idx)
_gather(A::AbstractArray, idx) = NNlib.gather(A, batched_index(idx, size(A)[end]))

_scatter(aggr, E, xs::AbstractArray) = NNlib.scatter(aggr, E, xs)
_scatter(aggr, E, xs::AbstractArray, dstsize) = NNlib.scatter(aggr, E, xs; dstsize=dstsize)

function batched_index(idx::AbstractVector, batch_size::Integer)
    b = copyto!(similar(idx, 1, batch_size), collect(1:batch_size))
    return tuple.(idx, b)
end

aggregate(aggr::typeof(+), X) = vec(sum(X, dims=2))
aggregate(aggr::typeof(-), X) = -vec(sum(X, dims=2))
aggregate(aggr::typeof(*), X) = vec(prod(X, dims=2))
aggregate(aggr::typeof(/), X) = 1 ./ vec(prod(X, dims=2))
aggregate(aggr::typeof(max), X) = vec(maximum(X, dims=2))
aggregate(aggr::typeof(min), X) = vec(minimum(X, dims=2))
aggregate(aggr::typeof(mean), X) = vec(aggr(X, dims=2))

@non_differentiable batched_index(x...)
