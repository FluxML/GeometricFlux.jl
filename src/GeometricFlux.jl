module GeometricFlux

using Statistics: mean
using LinearAlgebra: Adjoint, norm, Transpose
using Reexport

using CUDA
using FillArrays: Fill
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell, @functor
using NNlib, NNlibCUDA
using GraphLaplacians
@reexport using GraphSignals
using Graphs
using Zygote

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
    generate_cluster

include("datasets.jl")

include("utils.jl")

include("layers/graphlayers.jl")
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
