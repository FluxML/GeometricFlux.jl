using GeometricFlux
using GeometricFlux.Datasets
using Clustering
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
using Test

cuda_tests = [
    "cuda/conv",
    "cuda/msgpass",
]

tests = [
    "layers/gn",
    "layers/msgpass",
    "layers/conv",
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
