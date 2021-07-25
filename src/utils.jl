import Base: +, -, *, /, broadcasted

## Top-k pooling
"""
    accumulated_edges(adj)
>>>>>>> orig/master

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
end

### TODO move these to GraphSignals ######
# @functor FeaturedGraph
# Zygote.@nograd normalized_laplacian, scaled_laplacian


# Used for untrainable parameters, like epsilon in GINConv when set to false.
# NOTE: Only works for scalars untrainable params.
mutable struct Untrainable{T <: Number}
    value::T
end

+(u::Untrainable, z::Number) = u.value+z
@adjoint +(u::Untrainable, z::Number) = u+z, _->(nothing, 1.0)
+(z::Number,u::Untrainable) = u.value+z
@adjoint +(z::Number, u::Untrainable) = z+u, _->(1.0, nothing)
+(u1::Untrainable,u2::Untrainable) = u1.value + u2.value
@adjoint +(u1::Untrainable, u2::Untrainable) = u1+u2, _->(nothing,nothing)

-(u::Untrainable, z::Number) = u.value-z
@adjoint -(u::Untrainable, z::Number) = u-z, _->(nothing, -1.0)
-(z::Number,u::Untrainable) = z-u.value
@adjoint -(z::Number, u::Untrainable) = z-u, _->(1.0, nothing)
-(u1::Untrainable,u2::Untrainable) = u1.value - u2.value
@adjoint -(u1::Untrainable, u2::Untrainable) = u1-u2, _->(nothing,nothing)

*(u::Untrainable, z::Number) = u.value*z
@adjoint *(u::Untrainable, z::Number) = u*z, _->(nothing, u.value)
*(z::Number,u::Untrainable) = z*u.value
@adjoint *(z::Number, u::Untrainable) = z*u, _->(u.value, nothing)
*(u1::Untrainable,u2::Untrainable) = u1.value - u2.value
@adjoint *(u1::Untrainable, u2::Untrainable) = u1*u2, _->(nothing,nothing)

/(u::Untrainable, z::Number) = u.value/z
@adjoint /(u::Untrainable, z::Number) = u/z, _->(nothing, -u.value/(z^2))
/(z::Number,u::Untrainable) = z/u.value
@adjoint /(z::Number, u::Untrainable) = z/u, _->(1/u.value, nothing)
/(u1::Untrainable,u2::Untrainable) = u1.value - u2.value
@adjoint /(u1::Untrainable, u2::Untrainable) = u1/u2, _->(nothing,nothing)

broadcasted(::typeof(+), a::Untrainable, b::AbstractArray) = 
    broadcasted(+, a.value, b)
@adjoint broadcasted(::typeof(+), a::Untrainable, b::AbstractArray) = 
    broadcasted(+, a, b), _->(nothing, nothing, ones(size(b)))

broadcasted(::typeof(+), a::AbstractArray, b::Untrainable) = 
    broadcasted(+, a, b.value)
@adjoint broadcasted(::typeof(+), a::AbstractArray, b::Untrainable) = 
    broadcasted(+, a, b), _->(nothing, ones(size(b)), nothing)

broadcasted(::typeof(-), a::Untrainable, b::AbstractArray) = 
    broadcasted(-, a.value, b)
@adjoint broadcasted(::typeof(-), a::Untrainable, b::AbstractArray) = 
    broadcasted(-, a, b), _->(nothing, nothing, -ones(size(b)))

broadcasted(::typeof(-), a::AbstractArray, b::Untrainable) = 
    broadcasted(-, a, b.value)
@adjoint broadcasted(::typeof(-), a::AbstractArray, b::Untrainable) = 
    broadcasted(-, a, b), _->(nothing, ones(size(a)), nothing)

*(a::Untrainable, b::AbstractArray) = a.value * b
@adjoint *(a::Untrainable, b::AbstractArray) = 
    a * b, _->(nothing, nothing, a.value * ones(size(b)))
*(a::AbstractArray, b::Untrainable) = a * b.value
@adjoint broadcasted(::typeof(*), a::AbstractArray, b::Untrainable) = 
    a * b, _->(nothing, b.value * ones(size(a)), nothing)

broadcasted(::typeof(/), a::Untrainable, b::AbstractArray) = 
    broadcasted(/, a.value, b)
@adjoint broadcasted(::typeof(/), a::Untrainable, b::AbstractArray) = 
    broadcasted(/, a, b), _->(nothing, nothing, a.value ./ (b.^2))
    
/(a::AbstractArray, b::Untrainable) = /(a, b.value)
@adjoint /(a::AbstractArray, b::Untrainable) = 
    a/b, _->(nothing, (1/b.value) * ones(size(a)), nothing)

=======
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

