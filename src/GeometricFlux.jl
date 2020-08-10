module GeometricFlux

using Statistics: mean
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: I, issymmetric, diagm, eigmax, norm, Adjoint, Diagonal, eigen, Symmetric

using FillArrays: Fill
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell
using Flux: @functor
using LightGraphs
using Requires
using ScatterNNlib
using Zygote
using ZygoteRules

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

    # layer/selector
    FeatureSelector,

    # operations/linalg
    degree_matrix,
    laplacian_matrix,
    normalized_laplacian,
    scaled_laplacian,

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
    adjacency_list,
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

    # utils
    topk_index

const IntOrTuple = Union{Integer,Tuple}

include("operations/pool.jl")
include("operations/linalg.jl")

include("graph/index.jl")
include("graph/featuredgraphs.jl")
include("graph/linalg.jl")

include("utils.jl")

include("layers/gn.jl")
include("layers/msgpass.jl")

include("layers/conv.jl")
include("layers/pool.jl")
include("models.jl")
include("layers/selector.jl")

include("graph/simplegraphs.jl")


function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        using CUDA
        include("cuda/msgpass.jl")
        include("cuda/conv.jl")
        include("cuda/pool.jl")
    end
    @require SimpleWeightedGraphs = "47aef6b3-ad0c-573a-a1e2-d07658019622" begin
        include("graph/weightedgraphs.jl")
    end
    @require MetaGraphs = "626554b9-1ddb-594c-aa3c-2596fe9399a5" begin
        include("graph/metagraphs.jl")
    end
end

end
