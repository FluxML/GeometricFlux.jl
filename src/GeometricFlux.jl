module GeometricFlux

using Statistics: mean
using StaticArrays: StaticArray
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: I, issymmetric, diagm, eigmax, norm, Adjoint, Diagonal, Symmetric

using Requires
using DataStructures: DefaultDict
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell
using Flux: @functor
using LightGraphs
using Zygote
using ZygoteRules
using FillArrays: Fill

import Flux: maxpool, meanpool
import LightGraphs: nv, ne, adjacency_matrix

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

    # models
    GAE,
    VGAE,
    InnerProductDecoder,
    VariationalEncoder,

    # operations/linalg
    degree_matrix,
    laplacian_matrix,
    normalized_laplacian,
    scaled_laplacian,

    # operations/scatter
    scatter_add!,
    scatter_sub!,
    scatter_max!,
    scatter_min!,
    scatter_mul!,
    scatter_div!,
    scatter_mean!,
    scatter!,

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
    adjacency_list,
    generate_cluster,

    # graph/featuredgraphs
    AbstractFeaturedGraph,
    NullGraph,
    FeaturedGraph,
    graph,
    node_feature,
    edge_feature,
    global_feature,
    has_graph,
    has_node_feature,
    has_edge_feature,
    has_global_feature,
    nv,

    # utils
    gather,
    topk_index

using CUDAapi
if has_cuda()
    try
        using CuArrays
        @eval has_cuarrays() = true
    catch ex
        @warn "CUDA is installed, but CuArrays.jl fails to load" exception=(ex,catch_backtrace())
        @eval has_cuarrays() = false
    end
else
    has_cuarrays() = false
end

const IntOrTuple = Union{Integer,Tuple}

include("operations/scatter.jl")
include("operations/pool.jl")
include("operations/linalg.jl")

include("utils.jl")

include("graph/index.jl")
include("graph/featuredgraphs.jl")
include("graph/linalg.jl")

include("layers/gn.jl")
include("layers/msgpass.jl")

include("layers/conv.jl")
include("layers/pool.jl")
include("models.jl")

include("graph/simplegraphs.jl")


function __init__()
    @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        using CUDAnative
        using CuArrays: CuArray, CuMatrix, CuVector
        import CuArrays: cu
        include("cuda/scatter.jl")
        include("cuda/msgpass.jl")
        include("cuda/conv.jl")
        include("cuda/pool.jl")
        include("cuda/utils.jl")
        CuArrays.cu(x::Array{<:Integer}) = CuArray(x)
    end
    @require SimpleWeightedGraphs = "47aef6b3-ad0c-573a-a1e2-d07658019622" begin
        include("graph/weightedgraphs.jl")
    end
    @require MetaGraphs = "626554b9-1ddb-594c-aa3c-2596fe9399a5" begin
        include("graph/metagraphs.jl")
    end
end

end
