using LinearAlgebra
using SparseArrays: SparseMatrixCSC
using Statistics: mean
using Test

using GeometricFlux
using GeometricFlux.Datasets
using Clustering
using CUDA
using Flux
using Flux: @functor
using FillArrays
using GraphSignals
using Graphs
using NNlib

cuda_tests = [
    "cuda/graph_conv",
    "cuda/msgpass",
]

tests = [
    "layers/gn",
    "layers/msgpass",
    "layers/positional",
    "layers/graph_conv",
    "layers/group_conv",
    "layers/pool",
    "layers/graphlayers",
    "sampling",
    "models",
]

if CUDA.functional()
    append!(tests, cuda_tests)
else
    @warn "CUDA unavailable, not testing GPU support"
end

if !Sys.iswindows()
    push!(tests, "embedding/node2vec")
end

@testset "GeometricFlux" begin
    for t in tests
        include("$(t).jl")
    end
end
