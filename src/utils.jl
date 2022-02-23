function batched_index(idx::AbstractVector, batch_size::Integer)
    b = copyto!(similar(idx, 1, batch_size), collect(1:batch_size))
    return tuple.(idx, b)
end

batched_gather(::Nothing, idx) = nothing
batched_gather(A::Fill{T,2,Axes}, idx) where {T,Axes} = fill(A.value, A.axes[1], length(idx))
batched_gather(A::AbstractArray, idx) =
    NNlib.gather(A, batched_index(idx, size(A)[end]))

aggregate(aggr::typeof(+), X) = vec(sum(X, dims=2))
aggregate(aggr::typeof(-), X) = -vec(sum(X, dims=2))
aggregate(aggr::typeof(*), X) = vec(prod(X, dims=2))
aggregate(aggr::typeof(/), X) = 1 ./ vec(prod(X, dims=2))
aggregate(aggr::typeof(max), X) = vec(maximum(X, dims=2))
aggregate(aggr::typeof(min), X) = vec(minimum(X, dims=2))
aggregate(aggr::typeof(mean), X) = vec(aggr(X, dims=2))

function all_edges(fg::AbstractFeaturedGraph)
    es = GraphSignals.incident_edges(fg)
    xs = GraphSignals.repeat_nodes(fg)
    nbrs = GraphSignals.neighbors(fg)
    return es, xs, nbrs
end

@non_differentiable batched_index(x...)
@non_differentiable all_edges(x...)
