@testset "cuda/msgpass" begin
    T = Float32
    in_channel = 10
    out_channel = 5
    N = 6

    adj = [0. 1. 0. 0. 0. 0.;
        1. 0. 0. 1. 1. 1.;
        0. 0. 0. 0. 0. 1.;
        0. 1. 0. 0. 1. 0.;
        0. 1. 0. 1. 0. 1.;
        0. 1. 1. 0. 1. 0.]

    struct NewCudaLayer{T} <: MessagePassing
        weight::T
    end
    NewCudaLayer(m, n) = NewCudaLayer(randn(T, m,n))
    @functor NewCudaLayer

    function (l::NewCudaLayer)(fg::FeaturedGraph, X::AbstractMatrix)
        _, x, _ = GeometricFlux.propagate(l, fg, edge_feature(fg), X, global_feature(fg), +)
        x
    end

    (l::NewCudaLayer)(fg::FeaturedGraph) = FeaturedGraph(fg, nf = l(fg, node_feature(fg)))

    GeometricFlux.message(n::NewCudaLayer, x_i, x_j, e_ij) = n.weight * x_j
    GeometricFlux.update(::NewCudaLayer, m, x) = m

    X = rand(T, in_channel, N)
    fg = FeaturedGraph(adj, nf=X) |> gpu
    l = NewCudaLayer(out_channel, in_channel) |> gpu

    fg_ = l(fg)
    @test size(node_feature(fg_)) == (out_channel, N)
end
