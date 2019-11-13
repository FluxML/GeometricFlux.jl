using GeometricFlux
using GeometricFlux: neighbors, laplacian_matrix
using LightGraphs: SimpleGraph, SimpleDiGraph, add_edge!
using SimpleWeightedGraphs: SimpleWeightedGraph, SimpleWeightedDiGraph, add_edge!
using MetaGraphs: MetaGraph, MetaDiGraph
using Zygote
using Test

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

cuda_tests = [
    "cuda/scatter",
    "cuda/pool",
    "cuda/grad",
    "cuda/conv",
    "cuda/msgpass",
]

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
    "graph/metagraphs",
    "graph/utils",
    "utils",
]

if has_cuarrays()
    using Flux: gpu
    append!(tests, cuda_tests)
end

@testset "GeometricFlux" begin
    for t in tests
        include("$(t).jl")
    end
end
