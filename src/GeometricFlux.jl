module GeometricFlux
using Requires

using Core.Intrinsics: llvmcall
using Base.Threads
using Statistics: mean
using Flux: param, glorot_uniform, TrackedArray, leakyrelu, GRU
using Flux.Tracker: TrackedReal
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: I, issymmetric, diagm, eigmax

import Base.Threads: atomictypes, llvmtypes, inttype, ArithmeticTypes, FloatTypes,
       atomic_cas!,
       atomic_xchg!,
       atomic_add!, atomic_sub!,
       atomic_and!, atomic_nand!, atomic_or!, atomic_xor!,
       atomic_max!, atomic_min!

import Base.Sys: ARCH, WORD_SIZE

export

    # layers
    MessagePassing,
    GCNConv,
    ChebConv,
    GraphConv,
    GATConv,
    message,
    update,
    propagate,

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
    scatter_div!

include("scatter.jl")
include("layers.jl")
include("linalg.jl")


function __init__()
    @require LightGraphs = "093fc24a-ae57-5d10-9952-331d41423f4d" include("graph/simplegraphs.jl")
    @require SimpleWeightedGraphs = "47aef6b3-ad0c-573a-a1e2-d07658019622" include("graph/weightedgraphs.jl")
    @require MetaGraphs = "626554b9-1ddb-594c-aa3c-2596fe9399a5" include("metagraphs.jl")
end

end
