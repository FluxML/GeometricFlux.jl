using GeometricFlux
using GeometricFlux.Datasets
using CUDA
using Flux
using Flux: @functor
using FillArrays
using GraphSignals
using Graphs
using LinearAlgebra
using NNlib, NNlibCUDA
using SparseArrays: SparseMatrixCSC
using Statistics: mean
using Zygote
using Test
using Clustering

cuda_tests = [
    "cuda/conv",
    "cuda/msgpass",
]

tests = [
    "layers/gn",
    "layers/msgpass",
    "layers/conv",
    "layers/pool",
    "layers/misc",
    "graph_embedding/alias_sampling",
    "graph_embedding/node2vec",
    "models",
]

if CUDA.functional()
    append!(tests, cuda_tests)
else
    @warn "CUDA unavailable, not testing GPU support"
end

@testset "GeometricFlux" begin
    for t in tests
        include("$(t).jl")
    end
end
