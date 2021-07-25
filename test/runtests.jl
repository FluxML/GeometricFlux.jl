using GeometricFlux
using GeometricFlux.Datasets
using Flux
using Flux: @functor
using FillArrays
using GraphSignals
using LightGraphs: SimpleGraph, SimpleDiGraph, add_edge!, nv, ne
using SimpleWeightedGraphs: SimpleWeightedGraph, SimpleWeightedDiGraph, add_edge!
using SparseArrays: SparseMatrixCSC
using MetaGraphs: MetaGraph, MetaDiGraph
using Zygote
using Test

cuda_tests = [
#    "cuda/pool",
#    "cuda/grad",
#    "cuda/conv",
#    "cuda/msgpass",
]

tests = [
    "layers/gn",
    "layers/msgpass",
    "layers/conv",
    "layers/pool",
    "layers/selector",
    "grad",
    "models",
    "pool",
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
