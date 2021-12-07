module GeometricFlux

using Statistics: mean
using LinearAlgebra: Adjoint, norm, Transpose
using Reexport

using CUDA
using ChainRulesCore: @non_differentiable
using FillArrays: Fill
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell, @functor
using NNlib, NNlibCUDA
@reexport using GraphSignals
using Graphs
using Random
using Zygote
using SparseArrays
using DelimitedFiles

import Graphs: neighbors, is_directed, has_edge
import Word2Vec: word2vec, wordvectors, get_vector

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
    summarize,
    sample,

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

include("graph_embedding/sampling.jl")
include("graph_embedding/node2vec.jl")

include("cuda/conv.jl")

using .Datasets


end
