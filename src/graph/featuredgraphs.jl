abstract type AbstractFeaturedGraph end

"""
    NullGraph()

Null object for `FeaturedGraph`.
"""
struct NullGraph <: AbstractFeaturedGraph end

"""
    FeaturedGraph(graph, node_feature, edge_feature, global_feature)

A feature-equipped graph structure for passing graph to layer in order to provide graph dynamically.
References to graph or features are hold in this type.

# Arguments
- `graph`: should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `node_feature`: node features attached to graph.
- `edge_feature`: edge features attached to graph.
- `gloabl_feature`: gloabl graph features attached to graph.
"""
struct FeaturedGraph{T,S,R,Q} <: AbstractFeaturedGraph
    graph::Ref{T}
    nf::Ref{S}
    ef::Ref{R}
    gf::Ref{Q}

    function FeaturedGraph(graph::T, nf::S, ef::R, gf::Q) where {T,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
        new{T,S,R,Q}(Ref(graph), Ref(nf), Ref(ef), Ref(gf))
    end
end

FeaturedGraph() = FeaturedGraph(zeros(0,0), zeros(0,0), zeros(0,0), zeros(0))

FeaturedGraph(graph::T) where {T} = FeaturedGraph(graph, zeros(0,0), zeros(0,0), zeros(0))

FeaturedGraph(graph::T, nf::AbstractMatrix) where {T} = FeaturedGraph(graph, nf, zeros(0,0), zeros(0))

"""
    graph(::AbstractFeaturedGraph)

Get referenced graph.
"""
graph(::NullGraph) = nothing
graph(fg::FeaturedGraph) = fg.graph[]

"""
    node_feature(::AbstractFeaturedGraph)

Get node feature attached to graph.
"""
node_feature(::NullGraph) = nothing
node_feature(fg::FeaturedGraph) = fg.nf[]

"""
    edge_feature(::AbstractFeaturedGraph)

Get edge feature attached to graph.
"""
edge_feature(::NullGraph) = nothing
edge_feature(fg::FeaturedGraph) = fg.ef[]

"""
    global_feature(::AbstractFeaturedGraph)

Get global feature attached to graph.
"""
global_feature(::NullGraph) = nothing
global_feature(fg::FeaturedGraph) = fg.gf[]

has_graph(::NullGraph) = false
has_graph(fg::FeaturedGraph) = fg.graph[] != zeros(0,0)

has_node_feature(::NullGraph) = false
has_node_feature(fg::FeaturedGraph) = fg.nf[] != zeros(0,0)

has_edge_feature(::NullGraph) = false
has_edge_feature(fg::FeaturedGraph) = fg.ef[] != zeros(0,0)

has_global_feature(::NullGraph) = false
has_global_feature(fg::FeaturedGraph) = fg.gf[] != zeros(0)

"""
    neighbors(::AbstractFeaturedGraph)

Get adjacency list of graph.
"""
neighbors(::NullGraph) = [zeros(0)]
neighbors(fg::FeaturedGraph) = neighbors(fg.graph[])

"""
    nv(::AbstractFeaturedGraph)

Get node number of graph.
"""
nv(::NullGraph) = 0
nv(fg::FeaturedGraph) = nv(fg.graph[])
nv(fg::FeaturedGraph{T}) where {T<:AbstractMatrix} = size(fg.graph[], 1)

"""
    ne(::AbstractFeaturedGraph)

Get edge number of graph.
"""
ne(::NullGraph) = 0
ne(fg::FeaturedGraph) = ne(fg.graph[])



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
