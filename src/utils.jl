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


function gather(input::Matrix{T}, index::Array{Int}) where T
    out = Array{T}(undef, size(input,1), size(index)...)
    @inbounds for ind = CartesianIndices(index)
        out[:, ind] = input[:, index[ind]]
    end
    return out
end

identity(; kwargs...) = kwargs.data

struct GraphInfo
    adj::AbstractVector{<:AbstractVector}
    edge_idx::AbstractVector{<:Integer}
    V::Integer
    E::Integer

    function GraphInfo(adj)
        edge_idx = edge_index_table(adj)
        V = length(adj)
        E = edge_idx[end]
        new(adj, edge_idx, V, E)
    end
end

edge_index_table(adj::AbstractVector{<:AbstractVector}) = append!([0], cumsum(map(length, adj)))
