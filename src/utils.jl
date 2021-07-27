"""
    accumulated_edges(adj)

Return a vector which acts as a mapping table. The index is the vertex index,
value is accumulated numbers of edge (current vertex not included).
"""
accumulated_edges(adj::AbstractVector{<:AbstractVector{<:Integer}}) = [0, cumsum(map(length, adj))...]

Zygote.@nograd accumulated_edges

Zygote.@nograd function generate_cluster(M::AbstractArray{T,N}, accu_edge) where {T,N}
    num_V = length(accu_edge) - 1
    num_E = accu_edge[end]
    cluster = similar(M, Int, num_E)
    @inbounds for i = 1:num_V
        j = accu_edge[i]
        k = accu_edge[i+1]
        cluster[j+1:k] .= i
    end
    cluster
end

"""
    vertex_pair_table(adj[, num_E])

Generate a mapping from edge index to vertex pair (i, j). The edge indecies are determined by
the sorted vertex indecies.
"""
function vertex_pair_table(adj::AbstractVector{<:AbstractVector{<:Integer}},
                           num_E=sum(map(length, adj)))
    table = similar(adj[1], Tuple{UInt32,UInt32}, num_E)
    e = one(UInt64)
    for (i, js) = enumerate(adj)
        js = sort(js)
        for j = js
            table[e] = (i, j)
            e += one(UInt64)
        end
    end
    table
end

function vertex_pair_table(eidx::Dict)
    table = Array{Tuple{UInt32,UInt32}}(undef, num_E)
    for (k, v) = eidx
        table[v] = k
    end
    table
end

Zygote.@nograd vertex_pair_table

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

edge_index_table(fg::FeaturedGraph) = edge_index_table(fg.graph, fg.directed)

Zygote.@nograd edge_index_table

function transform(X::AbstractArray, vpair::AbstractVector{<:Tuple}, num_V)
    dims = size(X)[1:end-1]..., num_V, num_V
    Y = similar(X, dims)
    for (i, p) in enumerate(vpair)
        view(Y, :, p[1], p[2]) .= view(X, :, i)
    end
    Y
end

function transform(X::AbstractArray, eidx::Dict)
    dims = size(X)[1:end-2]..., length(eidx)
    Y = similar(X, dims)
    for (k, v) in eidx
        view(Y, :, v) .= view(X, :, k[1], k[2])
    end
    Y
end

### TODO move these to GraphSignals ######
# @functor FeaturedGraph
# Zygote.@nograd normalized_laplacian, scaled_laplacian
