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

function l2normalize(X::AbstractArray; dims=1)
    l2norm = .√(sum(abs2, X, dims=dims))
    return X ./ l2norm
end

function incidence_matrix(xs::AbstractVector{T}, N) where {T}
    A = similar(xs, T, size(xs, 1), N)
    copyto!(A, Array(I(N))[Array(xs), :])
    return A
end

function indexed_softmax(x::AbstractArray, xs, N; dims=1)
    y = copy(x)
    for i in 1:N
        idx = ntuple(j -> (j == dims) ? (xs .== i) : Colon(), ndims(y))
        NNlib.softmax!(view(y, idx...); dims)
    end
    return y
end

function ∇indexed_softmax(dy::AbstractArray{T}, y::AbstractArray{S}, xs, N; dims=1) where {T,S}
    dx = if NNlib.within_gradient(y)
        tmp = dy .* y
        for i in 1:N
            idx = ntuple(j -> (j == dims) ? (xs .== i) : Colon(), ndims(y))
            tmp[idx...] .= tmp[idx...] .- y[idx...] .* sum(tmp[idx...]; dims)
        end
        tmp
    else
        out = similar(y, promote_type(T,S))
        out .= dy .* y
        for i in 1:N
            idx = ntuple(j -> (j == dims) ? (xs .== i) : Colon(), ndims(y))
            out[idx...] .= out[idx...] .- y[idx...] .* sum(out[idx...]; dims)
        end
        out
    end
end

function ChainRulesCore.rrule(::typeof(indexed_softmax), x, xs, N; dims=1)
    y = indexed_softmax(x, xs, N; dims)
    indexed_softmax_pullback(dy) = (NoTangent(), ∇indexed_softmax(unthunk(dy), y, xs, N; dims), NoTangent(), NoTangent())
    return y, indexed_softmax_pullback
end

@non_differentiable batched_index(x...)
@non_differentiable incidence_matrix(x...)
