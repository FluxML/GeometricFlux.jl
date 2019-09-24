module GeometricFlux
using Requires

using Core.Intrinsics: llvmcall
using Base.Threads
using Statistics: mean
using DataStructures: DefaultDict
using Flux
using Flux: param, glorot_uniform, leakyrelu, GRUCell
using Flux: @treelike
using Flux.Tracker: TrackedArray, track, data, @grad
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: I, issymmetric, diagm, eigmax

import Base.Threads: atomictypes, llvmtypes, inttype, ArithmeticTypes, FloatTypes,
       atomic_cas!,
       atomic_xchg!,
       atomic_add!, atomic_sub!,
       atomic_and!, atomic_nand!, atomic_or!, atomic_xor!,
       atomic_max!, atomic_min!

import Base.Sys: ARCH, WORD_SIZE
import Flux: children

export

    # layers/msgpass
    MessagePassing,
    neighboring,

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
    sumpool,
    subpool,
    prodpool,
    divpool,
    maxpool,
    minpool,
    meanpool,
    pool,

    # models
    GAE,
    VGAE,
    InnerProductDecoder,
    VariationalEncoder,

    # linalg
    degree_matrix,
    laplacian_matrix,
    normalized_laplacian,
    neighbors,

    # scatter
    scatter_add!,
    scatter_sub!,
    scatter_max!,
    scatter_min!,
    scatter_mul!,
    scatter_div!,
    scatter_mean!,
    scatter!,

    # graph/utils
    adjlist,

    # utils
    gather

include("scatter.jl")
include("layers/msgpass.jl")
include("layers/conv.jl")
include("layers/pool.jl")
include("models.jl")
include("linalg.jl")
include("utils.jl")


function __init__()
    @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        using CUDAnative
        using CuArrays
        include("cuda/scatter.jl")
        include("cuda/pool.jl")
    end
    @require LightGraphs = "093fc24a-ae57-5d10-9952-331d41423f4d" begin
        include("graph/simplegraphs.jl")
    end
    @require SimpleWeightedGraphs = "47aef6b3-ad0c-573a-a1e2-d07658019622" begin
        include("graph/weightedgraphs.jl")
        include("graph/utils.jl")
    end
end

end
