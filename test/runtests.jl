using GeometricFlux
using GeometricFlux: neighbors, laplacian_matrix
using Flux
using Flux: @functor
using StaticArrays: @MMatrix, @MArray
using LightGraphs: SimpleGraph, SimpleDiGraph, add_edge!
using SimpleWeightedGraphs: SimpleWeightedGraph, SimpleWeightedDiGraph, add_edge!
using MetaGraphs: MetaGraph, MetaDiGraph
using Zygote
using Test

cuda_tests = [
    "cuda/scatter",
    "cuda/pool",
    "cuda/grad",
    "cuda/conv",
    "cuda/msgpass",
]

tests = [
    "layers/gn",
    "layers/msgpass",
    "layers/conv",
    "layers/pool",
    "grad",
    "models",
    "operations/linalg",
    "operations/scatter",
    "operations/pool",
    "graph/featuredgraphs",
    "graph/simplegraphs",
    "graph/weightedgraphs",
    "graph/metagraphs",
    "utils",
]

if Flux.use_cuda[]
    using CUDA
    using Flux: gpu
    append!(tests, cuda_tests)
else
    @warn "CUDA unavailable, not testing GPU support"
end

@testset "GeometricFlux" begin
    for t in tests
        include("$(t).jl")
    end
end
