module GeometricFlux

using Statistics: mean
using StaticArrays: StaticArray
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: I, issymmetric, diagm, eigmax, norm, Adjoint

using Requires
using DataStructures: DefaultDict
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell
using Flux: @functor
using ZygoteRules
using FillArrays: Fill

export

    # layers/meta
    Meta,
    adjlist,
    update_edge,
    update_vertex,
    update_global,
    aggregate_neighbors,
    aggregate_edges,
    aggregate_vertices,
    all_vertices_data,
    all_edges_data,
    adjacent_vertices_data,
    incident_edges_data,
    propagate,
    generate_cluster,

    # layers/msgpass
    MessagePassing,

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

    # graph/featuredgraphs
    AbstractFeaturedGraph,
    NullGraph,
    FeaturedGraph,
    graph,
    feature,

    # graph/utils
    adjlist,

    # utils
    gather,
    identity,
    GraphInfo,
    edge_index_table,
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

include("scatter.jl")
include("linalg.jl")
include("graph/featuredgraphs.jl")
include("utils.jl")
include("layers/meta.jl")
include("layers/msgpass.jl")
include("layers/conv.jl")
include("layers/pool.jl")
include("models.jl")


function __init__()
    @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        using CUDAnative
        using CuArrays: CuArray, CuMatrix, CuVector
        import CuArrays: cu
        include("cuda/scatter.jl")
        include("cuda/msgpass.jl")
        include("cuda/pool.jl")
        include("cuda/utils.jl")
        CuArrays.cu(x::Array{<:Integer}) = CuArray(x)
    end
    @require LightGraphs = "093fc24a-ae57-5d10-9952-331d41423f4d" begin
        include("graph/simplegraphs.jl")
    end
    @require SimpleWeightedGraphs = "47aef6b3-ad0c-573a-a1e2-d07658019622" begin
        include("graph/weightedgraphs.jl")
        include("graph/utils.jl")
    end
    @require MetaGraphs = "626554b9-1ddb-594c-aa3c-2596fe9399a5" begin
        include("graph/metagraphs.jl")
    end
end

end
