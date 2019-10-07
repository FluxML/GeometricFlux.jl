gather(input::AbstractArray{T}, index::CuArray{Int}) where T = gather(cu(input), index)

function gather(input::CuMatrix{T}, index::CuArray{Int}) where T
    out = CuArrays.zeros(T, size(input,1), size(index)...)
    @inbounds for ind = CartesianIndices(index)
        out[:, ind] = input[:, index[ind]]
    end
    return out
end
