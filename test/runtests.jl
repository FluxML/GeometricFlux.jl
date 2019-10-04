using GeometricFlux
using GeometricFlux: neighbors, laplacian_matrix
using LightGraphs: SimpleGraph, SimpleDiGraph, add_edge!
using SimpleWeightedGraphs: SimpleWeightedGraph, SimpleWeightedDiGraph, add_edge!
using Zygote
using Test

tests = [
    "cuda/scatter",
    "cuda/pool",
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
