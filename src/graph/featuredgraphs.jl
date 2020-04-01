abstract type AbstractFeaturedGraph end

struct NullGraph <: AbstractFeaturedGraph end

struct FeaturedGraph{T,S} <: AbstractFeaturedGraph
    graph::Ref{T}
    feature::Ref{S}
    FeaturedGraph(graph::T, feature::S) where {T,S} = new{T,S}(Ref(graph), Ref(feature))
end

graph(::NullGraph) = nothing
graph(fg::FeaturedGraph) = fg.graph[]

feature(::NullGraph) = nothing
feature(fg::FeaturedGraph) = fg.feature[]


## Linear algebra API for AbstractFeaturedGraph

"""
    degrees(g[, T; dir=:out])

Degree of each vertex. Return a vector which contains the degree of each vertex in graph `g`.

# Arguments
- `g`: should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `dir`: direction of degree; should be `:in`, `:out`, or `:both` (optional).

# Examples
```jldoctest
julia> using LightGraphs

julia> g = DiGraph(3);

julia> add_edge!(g, 2, 3);

julia> add_edge!(g, 3, 1);

julia> outdegree(g)
3-element Array{Int64,1}:
 0
 1
 1
```
"""
function degrees(fg::FeaturedGraph, T::DataType=eltype(fg.graph[]); dir::Symbol=:out)
    degrees(fg.graph[], T; dir=dir)
end

function degree_matrix(fg::FeaturedGraph, T::DataType=eltype(fg.graph[]); dir::Symbol=:out)
    degree_matrix(fg.graph[], T; dir=dir)
end

function inv_sqrt_degree_matrix(fg::FeaturedGraph, T::DataType=eltype(fg.graph[]); dir::Symbol=:out)
    inv_sqrt_degree_matrix(fg.graph[], T; dir=dir)
end

function laplacian_matrix(fg::FeaturedGraph, T::DataType=eltype(fg.graph[]); dir::Symbol=:out)
    laplacian_matrix(fg.graph[], T; dir=dir)
end

function normalized_laplacian(fg::FeaturedGraph, T::DataType=eltype(fg.graph[]))
    normalized_laplacian(fg.graph[], T)
end


## Convolution layers accepting AbstractFeaturedGraph

# function (g::GCNConv)(gr::FeaturedGraph)
#     X = gr.feature[]
#     A = gr.graph[]
#     g.σ.(g.weight * X * normalized_laplacian(A+I) .+ g.bias)
# end
