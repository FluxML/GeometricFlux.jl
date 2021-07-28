module GeometricFlux

using Base: Tuple
using Statistics: mean
using LinearAlgebra
using FillArrays: Fill

using CUDA
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell, @functor
using NNlib, NNlibCUDA
using Zygote
using ChainRulesCore


# import GraphLaplacians
# using GraphLaplacians: normalized_laplacian, scaled_laplacian
# using GraphLaplacians: adjacency_matrix
# using Reexport
# @reexport using GraphSignals
import LightGraphs
using LightGraphs: AbstractGraph, outneighbors, inneighbors, is_directed, ne, nv, adjacency_matrix

export
    FeaturedGraph,
    adjacency_list,
    node_feature, edge_feature, global_feature,
    ne, nv, adjacency_matrix, # from LightGraphs

    # layers/gn
    GraphNet,

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

include("layers/gn.jl")
include("layers/msgpass.jl")

include("layers/conv.jl")
include("layers/pool.jl")
include("models.jl")
include("layers/misc.jl")

include("cuda/msgpass.jl")
include("cuda/conv.jl")

using .Datasets


end
