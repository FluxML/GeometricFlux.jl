_gather(::Nothing, idx) = nothing
_gather(A::Fill{T,2,Axes}, idx) where {T,Axes} = fill(A.value, A.axes[1], length(idx))
_gather(A::AbstractMatrix, idx) = NNlib.gather(A, idx)
_gather(A::AbstractArray, idx) = NNlib.gather(A, batched_index(idx, size(A)[end]))

_scatter(aggr, E, xs::AbstractArray) = NNlib.scatter(aggr, E, xs)
_scatter(aggr, E, xs::AbstractArray, dstsize) = NNlib.scatter(aggr, E, xs; dstsize=dstsize)

_matmul(A::AbstractMatrix, B::AbstractMatrix) = A * B
_matmul(A::AbstractArray, B::AbstractArray) = NNlib.batched_mul(A, B)

function batched_index(idx::AbstractVector, batch_size::Integer)
    b = copyto!(similar(idx, 1, batch_size), collect(1:batch_size))
    return tuple.(idx, b)
end

aggregate(::typeof(+), X) = sum(X, dims=2)
aggregate(::typeof(-), X) = -sum(X, dims=2)
aggregate(::typeof(*), X) = prod(X, dims=2)
aggregate(::typeof(/), X) = 1 ./ prod(X, dims=2)
aggregate(::typeof(max), X) = maximum(X, dims=2)
aggregate(::typeof(min), X) = minimum(X, dims=2)
aggregate(::typeof(mean), X) = mean(X, dims=2)

@non_differentiable batched_index(x...)
