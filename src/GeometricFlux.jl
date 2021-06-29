module GeometricFlux

using Statistics: mean
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: Adjoint, norm, Transpose
using Reexport

using CUDA
using FillArrays: Fill
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell
using Flux: @functor
@reexport using GraphSignals
using LightGraphs
using Requires
using ScatterNNlib
using Zygote
using ZygoteRules

import Flux: maxpool, meanpool

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
    GINConv,
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
    summarize,
    sample,

    # layer/selector
    FeatureSelector,

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
    generate_cluster,

    # utils
    topk_index,
    bypass_graph

const IntOrTuple = Union{Integer,Tuple}

include("datasets.jl")

include("pool.jl")

include("graph/index.jl")

include("utils.jl")

include("layers/gn.jl")
include("layers/msgpass.jl")

include("layers/conv.jl")
include("layers/pool.jl")
include("models.jl")
include("layers/selector.jl")

include("graph/simplegraphs.jl")


using .Datasets

function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
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
