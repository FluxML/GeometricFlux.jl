function gather(input::AbstractArray{T,N}, index::AbstractArray{<:Integer,N}, dims::Integer;
                out::AbstractArray{T,N}=similar(index, T)) where {T,N}
    @assert dims <= N "Specified dimensions must lower or equal to the rank of input matrix."

    @inbounds for x = CartesianIndices(out)
        tup = collect(Tuple(x))
        tup[dims] = index[x]
        out[x] = input[tup...]
    end
    return out
end


function gather(input::AbstractMatrix{T}, index::AbstractArray{Int}) where T
    out = Array{T}(undef, size(input,1), size(index)...)
    for ind = CartesianIndices(index)
        k = index[ind]
        out[:, ind] = input[:, k]
    end
    return out
end
