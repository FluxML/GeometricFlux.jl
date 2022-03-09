@testset "cuda/msgpass" begin
    T = Float32
    in_channel = 10
    out_channel = 5
    N = 6
    batch_size = 10

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

    function (l::NewCudaLayer)(fg::AbstractFeaturedGraph)
        nf = node_feature(fg)
        GraphSignals.check_num_nodes(fg, nf)
        _, V, _ = GeometricFlux.propagate(l, graph(fg), nothing, nf, nothing, +, nothing, nothing)
        return FeaturedGraph(fg, nf=V)
    end

    # For static graph
    function (l::NewCudaLayer)(el::NamedTuple, x::AbstractArray)
        GraphSignals.check_num_nodes(el.N, size(x, 2))
        _, x, _ = GeometricFlux.propagate(l, el, nothing, x, nothing, +, nothing, nothing)
        return x
    end

    GeometricFlux.message(n::NewCudaLayer, x_i, x_j::AbstractMatrix, e_ij) = n.weight * x_j
    GeometricFlux.message(n::NewCudaLayer, x_i, x_j::AbstractArray, e_ij) = NNlib.batched_mul(n.weight, x_j)
    GeometricFlux.update(::NewCudaLayer, m, x) = m
    
    @testset "layer without graph" begin
        l = NewCudaLayer(out_channel, in_channel) |> gpu

        X = rand(T, in_channel, N)
        fg = FeaturedGraph(adj, nf=X) |> gpu
        fg_ = l(fg)
        @test size(node_feature(fg_)) == (out_channel, N)

        g = Zygote.gradient(() -> sum(node_feature(l(fg))), Flux.params(l))
        @test length(g.grads) == 3
    end

    @testset "layer with static graph" begin
        batch_size = 10
        X = rand(T, in_channel, N, batch_size)
        fg = FeaturedGraph(adj)
        l = WithGraph(fg, NewCudaLayer(out_channel, in_channel)) |> gpu
        Y = l(X |> gpu)
        @test size(Y) == (out_channel, N, batch_size)

        g = Zygote.gradient(() -> sum(l(X |> gpu)), Flux.params(l))
        @test length(g.grads) == 2
    end
end
