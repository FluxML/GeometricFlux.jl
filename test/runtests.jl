using GeometricFlux
using GeometricFlux.Datasets
using Flux
using Flux: @functor
using LightGraphs
using LinearAlgebra
using NNlib
using Statistics: mean
using Zygote
using Test

cuda_tests = [
    "cuda/conv",
    "cuda/msgpass",
]

tests = [
    "featured_graph",
    "layers/msgpass",
    "layers/conv",
    "layers/pool",
    "layers/misc",
    "models",
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
