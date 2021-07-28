"""
    accumulated_edges(adj)

Return a vector which acts as a mapping table. The index is the vertex index,
value is accumulated numbers of edge (current vertex not included).
"""
accumulated_edges(adj::AbstractVector{<:AbstractVector{<:Integer}}) = [0, cumsum(map(length, adj))...]

Zygote.@nograd accumulated_edges


Zygote.@nograd function generate_cluster(M, accu_edge)
    num_V = length(accu_edge) - 1
    num_E = accu_edge[end]
    # cluster = similar(M, Int, num_E)
    cluster = zeros(Int, num_E)
    @inbounds for i = 1:num_V
        j = accu_edge[i]
        k = accu_edge[i+1]
        cluster[j+1:k] .= i
    end
    cluster
end

"""
    edge_index_table(adj[, directed])

Generate a mapping from vertex pair (i, j) to edge index. The edge indecies are determined by
the sorted vertex indecies.
"""
function edge_index_table(adj::AbstractVector{<:AbstractVector{<:Integer}}, directed::Bool=is_directed(adj))
    table = Dict{Tuple{UInt32,UInt32},UInt64}()
    e = one(UInt64)
    if directed
        for (i, js) = enumerate(adj)
            js = sort(js)
            for j = js
                table[(i, j)] = e
                e += one(UInt64)
            end
        end
    else
        for (i, js) = enumerate(adj)
            js = sort(js)
            js = js[i .≤ js]
            for j = js
                table[(i, j)] = e
                table[(j, i)] = e
                e += one(UInt64)
            end
        end
    end
    table
end

function edge_index_table(vpair::AbstractVector{<:Tuple})
    table = Dict{Tuple{UInt32,UInt32},UInt64}()
    for (i, p) = enumerate(vpair)
        table[p] = i
    end
    table
end

Zygote.@nograd edge_index_table

function check_num_nodes(fg::FeaturedGraph, x::AbstractArray)
    @assert nv(fg) == size(x, ndims(x))    
end