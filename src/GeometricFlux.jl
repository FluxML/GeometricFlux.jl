module GeometricFlux

using NNlib: similar
using ChainRulesCore: eltype, reshape
using LinearAlgebra: similar
using Statistics: mean
using LinearAlgebra

using CUDA
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell, @functor
using NNlib, NNlibCUDA
using Zygote
using ChainRulesCore
import LightGraphs
using LightGraphs: AbstractGraph, outneighbors, inneighbors, is_directed, ne, nv, 
                  adjacency_matrix, degree

export
    # featured_graph
    FeaturedGraph,
    edge_index,
    node_feature, edge_feature, global_feature,
    adjacency_list, normalized_laplacian, scaled_laplacian,

    # from LightGraphs
    ne, nv, adjacency_matrix, 

    # layers/msgpass
    MessagePassing,

    # layers/conv
    GCNConv,
    ChebConv,
    GraphConv,
    GATConv,
    GatedGraphConv,
    EdgeConv,

    # layer/pool
    GlobalPool,
    LocalPool,
    TopKPool,
    topk_index,

    # models
    GAE,
    VGAE,
    InnerProductDecoder,
    VariationalEncoder,
    summarize,
    sample,

    # layer/selector
    bypass_graph

include("featured_graph.jl")  
include("datasets.jl")

include("utils.jl")

include("layers/msgpass.jl")

include("layers/conv.jl")
include("layers/pool.jl")
include("models.jl")
include("layers/misc.jl")

using .Datasets


end
