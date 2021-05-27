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
    topk_index,

    # models
    GAE,
    VGAE,
    InnerProductDecoder,
    VariationalEncoder,
    summarize,
    sample,

    # layer/selector
    FeatureSelector,
    bypass_graph,

    # utils
    generate_cluster

const IntOrTuple = Union{Integer,Tuple}

include("datasets.jl")

include("scatter.jl")
include("utils.jl")

include("layers/gn.jl")
include("layers/msgpass.jl")

include("layers/conv.jl")
include("layers/pool.jl")
include("models.jl")
include("layers/misc.jl")

include("graphs.jl")


using .Datasets

function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("cuda/scatter.jl")
        include("cuda/msgpass.jl")
        include("cuda/conv.jl")
    end
end

end
