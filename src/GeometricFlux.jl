module GeometricFlux
using Requires

using Flux: param, glorot_uniform
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: I, issymmetric, diagm, eigmax

export

    # layers
    # MessagePassing,
    GCNConv,
    ChebConv,
    GraphConv,
    # GATConv,
    message,
    update,

    # linalg
    degree_matrix,
    laplacian_matrix,
    normalized_laplacian,
    neighbors

include("layers.jl")
include("linalg.jl")

function __init__()
    @require LightGraphs = "093fc24a-ae57-5d10-9952-331d41423f4d" include("graph/simplegraphs.jl")
    @require SimpleWeightedGraphs = "47aef6b3-ad0c-573a-a1e2-d07658019622" include("graph/weightedgraphs.jl")
    @require MetaGraphs = "626554b9-1ddb-594c-aa3c-2596fe9399a5" include("metagraphs.jl")
end

end
