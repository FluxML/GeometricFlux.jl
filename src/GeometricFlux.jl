module GeometricFlux

using Statistics: mean
using LinearAlgebra: Adjoint, norm, Transpose
using Reexport

using CUDA
using FillArrays: Fill
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell
using Flux: @functor
using GraphLaplacians
@reexport using GraphSignals
using LightGraphs
using Requires
using Zygote

export
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
    FeatureSelector,
    bypass_graph,

    # utils
    generate_cluster

include("datasets.jl")

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
        using NNlibCUDA

        include("cuda/msgpass.jl")
        include("cuda/conv.jl")
    end
end

end
