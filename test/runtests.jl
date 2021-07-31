using GeometricFlux
using GeometricFlux.Datasets
using Flux
using Flux: @functor
using FillArrays
using LinearAlgebra
using NNlib
using LightGraphs
using Statistics: mean
using Zygote
using Test

cuda_tests = [
    # "cuda/featured_graph",
    # "cuda/conv",
    # "cuda/msgpass",
]

tests = [
    "featured_graph",
    # "layers/gn",
    # "layers/msgpass",
    # "layers/conv",
    # "layers/pool",
    # "layers/misc",
    # "models",
]

if Flux.use_cuda[]
    using CUDA
    using Flux: gpu
    using NNlibCUDA
    append!(tests, cuda_tests)
else
    @warn "CUDA unavailable, not testing GPU support"
end

@testset "GeometricFlux" begin
    for t in tests
        include("$(t).jl")
    end
end
