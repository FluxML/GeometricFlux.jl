abstract type AbstractFeaturedGraph end

"""
    NullGraph()

Null object for `FeaturedGraph`.
"""
struct NullGraph <: AbstractFeaturedGraph end

"""
    FeaturedGraph(graph, feature)

A feature-equipped graph structure for passing graph to layer in order to provide graph dynamically.
References to graph or feature are hold in this type. `graph` and `feature` are provided to fetch data.

# Arguments
- `graph`: should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `feature`: features attached to graph.
"""
struct FeaturedGraph{T,S} <: AbstractFeaturedGraph
    graph::Ref{T}
    feature::Ref{S}
    FeaturedGraph(graph::T, feature::S) where {T,S} = new{T,S}(Ref(graph), Ref(feature))
end

"""
    graph(::AbstractFeaturedGraph)

Get referenced graph.
"""
graph(::NullGraph) = nothing
graph(fg::FeaturedGraph) = fg.graph[]

"""
    feature(::AbstractFeaturedGraph)

Get feature attached to graph.
"""
feature(::NullGraph) = nothing
feature(fg::FeaturedGraph) = fg.feature[]

nv(::NullGraph) = 0
nv(fg::FeaturedGraph) = nv(fg.graph[])
nv(fg::FeaturedGraph{T}) where {T<:AbstractMatrix} = size(fg.graph[], 1)


## Linear algebra API for AbstractFeaturedGraph

adjacency_matrix(fg::FeaturedGraph, T::DataType=eltype(fg.graph[])) = adjacency_matrix(fg.graph[], T)

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

function normalized_laplacian(fg::FeaturedGraph, T::DataType=eltype(fg.graph[]); selfloop::Bool=false)
    normalized_laplacian(fg.graph[], T; selfloop=selfloop)
end

function scaled_laplacian(fg::FeaturedGraph, T::DataType=eltype(fg.graph[]))
    scaled_laplacian(fg.graph[], T)
end
