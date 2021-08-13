using GeometricFlux
using GeometricFlux.Datasets
using GeometricFlux: sort_edge_index
using Flux
using CUDA
using Flux: gpu
using Flux: @functor
using FillArrays
using LinearAlgebra
using NNlib
using LightGraphs
using Statistics: mean
using Zygote
using Test

cuda_tests = [
    "cuda/featured_graph",
    # "cuda/conv",
    # "cuda/msgpass",
]

tests = [
    # "featured_graph",
    # "layers/gn",
    # "layers/msgpass",
    # "layers/conv",
    # "layers/pool",
    # "layers/misc",
    # "models",
]

if Flux.use_cuda[]
    append!(tests, cuda_tests)
else
    @warn "CUDA unavailable, not testing GPU support"
end

# Testing all graph types. :sparse is a bit broken at the moment
@testset "GeometricFlux: graph format $graph_type" for graph_type in (:coo, :dense, :sparse)
    global GRAPH_T = graph_type
    for t in tests
        include("$(t).jl")
    end

    if Flux.use_cuda[] && GRAPH_T != :sparse
        for t in cuda_tests
            include("$(t).jl")
        end
    else
        @warn "CUDA unavailable, not testing GPU support"
    end
end
