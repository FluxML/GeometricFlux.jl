using GeometricFlux
using GeometricFlux: neighbors, laplacian_matrix
using LightGraphs
using LightGraphs: neighbors, laplacian_matrix
using SimpleWeightedGraphs
using Zygote
using Test

tests = [
    "layers/msgpass",
    "layers/conv",
    "layers/pool",
    "grad",
    "models",
    "linalg",
    "scatter",
    "graph/simplegraphs",
    "graph/weightedgraphs",
    "graph/utils",
    "utils",
]

@testset "GeometricFlux" begin
    for t in tests
        include("$(t).jl")
    end
end
