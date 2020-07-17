module GeometricFlux

using Statistics: mean
using StaticArrays: StaticArray
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: I, issymmetric, diagm, eigmax, norm, Adjoint, Diagonal

using Requires
using DataStructures: DefaultDict
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell
using Flux: @functor
using LightGraphs
using Zygote
using ZygoteRules
using FillArrays: Fill

import Flux: maxpool, meanpool
import LightGraphs: nv, ne, adjacency_matrix

export
    # layers/gn
    GraphNet,
    update_edge,
    update_vertex,
    update_global,
    update_batch_edge,
    update_batch_vertex,
    aggregate_neighbors,
    aggregate_edges,
    aggregate_vertices,
    propagate,

    # layers/msgpass
    MessagePassing,
    message,
    update,

    # layers/conv
    GCNConv,
    ChebConv,
    GraphConv,
    GATConv,
    GatedGraphConv,
    EdgeConv,
    message,
    update,
    propagate,

    # layer/pool
    GlobalPool,
    LocalPool,
    TopKPool,

    # models
    GAE,
    VGAE,
    InnerProductDecoder,
    VariationalEncoder,

    # operations/linalg
    degree_matrix,
    laplacian_matrix,
    normalized_laplacian,
    scaled_laplacian,

    # operations/scatter
    scatter_add!,
    scatter_sub!,
    scatter_max!,
    scatter_min!,
    scatter_mul!,
    scatter_div!,
    scatter_mean!,
    scatter!,

    # operations/pool
    sumpool,
    subpool,
    prodpool,
    divpool,
    maxpool,
    minpool,
    meanpool,
    pool,

    # graph/index
    neighbors,
    generate_cluster,

    # graph/featuredgraphs
    AbstractFeaturedGraph,
    NullGraph,
    FeaturedGraph,
    graph,
    node_feature,
    edge_feature,
    global_feature,
    has_graph,
    has_node_feature,
    has_edge_feature,
    has_global_feature,
    nv,

    # graph/simplegraphs
    adjlist,

    # utils
    gather,
    topk_index

const IntOrTuple = Union{Integer,Tuple}

include("operations/scatter.jl")
include("operations/pool.jl")
include("operations/linalg.jl")

include("utils.jl")

include("graph/index.jl")
include("graph/featuredgraphs.jl")
include("graph/linalg.jl")

include("layers/gn.jl")
include("layers/msgpass.jl")

include("layers/conv.jl")
include("layers/pool.jl")
include("models.jl")

include("graph/simplegraphs.jl")


function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        using CUDA: CuArray, CuMatrix, CuVector, CuDeviceArray
        using CUDA: @cuda
        import CUDA: cu
        include("cuda/scatter.jl")
        include("cuda/msgpass.jl")
        include("cuda/conv.jl")
        include("cuda/pool.jl")
        include("cuda/utils.jl")
        CUDA.cu(x::Array{<:Integer}) = CuArray(x)
    end
    @require SimpleWeightedGraphs = "47aef6b3-ad0c-573a-a1e2-d07658019622" begin
        include("graph/weightedgraphs.jl")
    end
    @require MetaGraphs = "626554b9-1ddb-594c-aa3c-2596fe9399a5" begin
        include("graph/metagraphs.jl")
    end
end

end
