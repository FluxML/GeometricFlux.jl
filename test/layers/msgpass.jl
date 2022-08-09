@testset "msgpass" begin
    T = Float32
    in_channel = 10
    out_channel = 5
    num_V = 6
    num_E = 7

    adj = T[0. 1. 0. 0. 0. 0.;
            1. 0. 0. 1. 1. 1.;
            0. 0. 0. 0. 0. 1.;
            0. 1. 0. 0. 1. 0.;
            0. 1. 0. 1. 0. 1.;
            0. 1. 1. 0. 1. 0.]

    struct NewLayer{T} <: MessagePassing
        weight::T
    end
    NewLayer(m, n) = NewLayer(randn(T, m, n))
    @functor NewLayer

    # For variable graph
    function (l::NewLayer)(fg::AbstractFeaturedGraph)
        nf = node_feature(fg)
        GraphSignals.check_num_nodes(fg, nf)
        _, V, _ = GeometricFlux.propagate(l, graph(fg), nothing, nf, nothing, +, nothing, nothing)
        return FeaturedGraph(fg, nf=V)
    end

    X = Array{T}(reshape(1:num_V*in_channel, in_channel, num_V))
    fg = FeaturedGraph(adj, nf=X, ef=Fill(zero(T), 0, num_E))

    l = NewLayer(out_channel, in_channel)

    @testset "no message or update" begin
        fg_ = l(fg)

        @test GraphSignals.adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (in_channel, num_V)
        @test size(edge_feature(fg_)) == (0, num_E)
        @test !has_global_feature(fg_)
    end

    GeometricFlux.message(l::NewLayer, x_i, x_j::AbstractMatrix, e_ij) = l.weight * x_j
    @testset "message function" begin
        fg_ = l(fg)

        @test GraphSignals.adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (out_channel, num_V)
        @test size(edge_feature(fg_)) == (0, num_E)
        @test !has_global_feature(fg_)
    end

    GeometricFlux.update(l::NewLayer, m::AbstractMatrix, x) = l.weight * x + m
    @testset "message and update" begin
        fg_ = l(fg)

        @test GraphSignals.adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (out_channel, num_V)
        @test size(edge_feature(fg_)) == (0, num_E)
        @test !has_global_feature(fg_)
    end
end
