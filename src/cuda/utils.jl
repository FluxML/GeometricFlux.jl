## Inverse operation of scatter

gather(input::AbstractArray{T}, index::CuArray{Int}) where T = gather(cu(input), index)

function gather(input::CuMatrix{T}, index::CuArray{Int}) where T
    out = CUDA.zeros(T, size(input,1), size(index)...)
    @inbounds for ind = CartesianIndices(index)
        out[:, ind] = input[:, index[ind]]
    end
    return out
end

function gather_indices(X::CuArray{T}) where T
    Y = gather_indices(Array(X))
    cuY = Dict{T,CuVector}(k => cu(Tuple.(v)) for (k, v) in Y)
    cuY
end

function numerical_cmp(X::CuArray{T}, Y::CuArray) where T
    Z = map((x,y) -> sign(x - y)^2, X, Y)
    Z = map(x -> (one(T) - x)^2, Z)
    Z
end
