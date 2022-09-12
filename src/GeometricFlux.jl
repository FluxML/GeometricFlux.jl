module GeometricFlux

using DelimitedFiles
using SparseArrays
using Statistics, StatsBase
using LinearAlgebra
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
using Optimisers

import Word2Vec: word2vec, wordvectors, get_vector

export
    # layers/graphlayers
    AbstractGraphLayer,

    # layers/gn
    GraphNet,

    # layers/msgpass
    MessagePassing,

    # layers/positional
    AbstractPositionalEncoding,
    RandomWalkPE,
    LaplacianPE,
    positional_encode,
    EEquivGraphPE,
    LSPE,
    GatedGCNLSPEConv,

    # layers/graphconv
    GCNConv,
    ChebConv,
    GraphConv,
    GATConv,
    GATv2Conv,
    GatedGraphConv,
    EdgeConv,
    GINConv,
    CGConv,
    SAGEConv,
    MeanAggregator, MeanPoolAggregator, MaxPoolAggregator,
    LSTMAggregator,
    GatedGCNConv,

    # layers/groupconv
    EEquivGraphConv,

    # layer/pool
    GlobalPool,
    LocalPool,
    TopKPool,
    topk_index,

    # models
    GAE,
    VGAE,
    InnerProductDecoder,
    VariationalGraphEncoder,
    DeepSet,

    # loss
    laplacian_eig_loss,

    # layer/utils
    WithGraph,
    GraphParallel,

    #node2vec
    node2vec

include("datasets.jl")
include("operation.jl")
include("loss.jl")

include("layers/graphlayers.jl")
include("layers/gn.jl")
include("layers/msgpass.jl")

include("layers/positional.jl")
include("layers/graphconv.jl")
include("layers/groupconv.jl")
include("layers/pool.jl")
include("models.jl")

include("sampling.jl")
include("embedding/node2vec.jl")

using .Datasets


end
