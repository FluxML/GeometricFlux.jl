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
