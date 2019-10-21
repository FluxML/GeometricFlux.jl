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

struct GraphInfo{A,T<:Integer}
    adj::AbstractVector{A}
    edge_idx::A
    V::T
    E::T

    function GraphInfo(adj::AbstractVector{<:AbstractVector{<:Integer}})
        V = length(adj)
        edge_idx = edge_index_table(adj, V)
        E = edge_idx[end]
        new{typeof(edge_idx),typeof(V)}(adj, edge_idx, V, E)
    end
end

function edge_index_table(adj::AbstractVector{<:AbstractVector{<:Integer}},
                          N::Integer=size(adj,1))
    y = similar(adj[1], N+1)
    y .= 0, cumsum(map(length, adj))...
    y
end
