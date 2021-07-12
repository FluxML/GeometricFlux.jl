in_channel = 10
out_channel = 5
num_V = 6
num_E = 7
T = Float32

adj = T[0. 1. 0. 0. 0. 0.;
       1. 0. 0. 1. 1. 1.;
       0. 0. 0. 0. 0. 1.;
       0. 1. 0. 0. 1. 0.;
       0. 1. 0. 1. 0. 1.;
       0. 1. 1. 0. 1. 0.]

struct NewLayer <: MessagePassing
    weight
end
NewLayer(m, n) = NewLayer(randn(T, m,n))

(l::NewLayer)(fg) = GeometricFlux.propagate(l, fg, +)

X = Array{T}(reshape(1:num_V*in_channel, in_channel, num_V))
fg = FeaturedGraph(adj, nf=X, ef=Fill(zero(T), 0, 2num_E))

l = NewLayer(out_channel, in_channel)

@testset "msgpass" begin
    @testset "no message or update" begin
        fg_ = l(fg)

        @test graph(fg_) == adj
        @test size(node_feature(fg_)) == (in_channel, num_V)
        @test size(edge_feature(fg_)) == (in_channel, 2*num_E)
        @test size(global_feature(fg_)) == (0,)
    end

    GeometricFlux.message(l::NewLayer, x_i, x_j, e_ij) = l.weight * x_j
    @testset "message function" begin
        fg_ = l(fg)

        @test graph(fg_) == adj
        @test size(node_feature(fg_)) == (out_channel, num_V)
        @test size(edge_feature(fg_)) == (out_channel, 2*num_E)
        @test size(global_feature(fg_)) == (0,)
    end

    GeometricFlux.update(l::NewLayer, m, x) = l.weight * x + m
    @testset "message and update" begin
        fg_ = l(fg)

        @test graph(fg_) == adj
        @test size(node_feature(fg_)) == (out_channel, num_V)
        @test size(edge_feature(fg_)) == (out_channel, 2*num_E)
        @test size(global_feature(fg_)) == (0,)
    end
end
