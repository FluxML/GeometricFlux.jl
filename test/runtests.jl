using GeometricFlux
using GeometricFlux: neighbors, laplacian_matrix
using LightGraphs
using LightGraphs: neighbors, laplacian_matrix
using SimpleWeightedGraphs
using Test

tests = [
    "layers/conv",
    "layers/pool",
    "linalg",
    "scatter",
    "graph/simplegraphs",
    "graph/weightedgraphs",
    "graph/utils"
]

@testset "GeometricFlux" begin
    for t in tests
        include("$(t).jl")
    end
end
