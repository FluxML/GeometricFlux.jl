module GeometricFlux

using NNlib: similar
using LinearAlgebra: similar, fill!
using Statistics: mean
using LinearAlgebra
using SparseArrays
import KrylovKit
using CUDA
using FillArrays: Fill
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell, @functor
using NNlib, NNlibCUDA
using ChainRulesCore
import LightGraphs
using LightGraphs: AbstractGraph, outneighbors, inneighbors, is_directed, ne, nv, 
                  adjacency_matrix, degree

export
    # featured_graph
    FeaturedGraph,
    graph, edge_index,
    node_feature, edge_feature, global_feature,
    adjacency_list, normalized_laplacian, scaled_laplacian,

    # from LightGraphs
    adjacency_matrix, 

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
    GINConv,

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
    bypass_graph,

    # utils
    generate_cluster

    
include("featuredgraph.jl")
include("graph_conversions.jl")
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
