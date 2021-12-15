module GeometricFlux

using DelimitedFiles
using SparseArrays
using Statistics: mean
using LinearAlgebra: Adjoint, norm, Transpose
using Random
using Reexport

using CUDA, CUDA.CUSPARSE
using ChainRulesCore
using ChainRulesCore: @non_differentiable
using FillArrays: Fill
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell, @functor
@reexport using GraphSignals
using Graphs
using NNlib, NNlibCUDA
using Zygote

import Word2Vec: word2vec, wordvectors, get_vector

const ConcreteFeaturedGraph = Union{FeaturedGraph,FeaturedSubgraph}

export
    # layers/graphlayers
    AbstractGraphLayer,

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
    CGConv,

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

    # layer/selector
    bypass_graph,

    # utils
    generate_cluster,

    #node2vec
    node2vec

include("datasets.jl")

include("utils.jl")

include("layers/graphlayers.jl")
include("layers/gn.jl")
include("layers/msgpass.jl")

include("layers/conv.jl")
include("layers/pool.jl")
include("models.jl")
include("layers/misc.jl")

include("sampling.jl")
include("embedding/node2vec.jl")

using .Datasets


end
