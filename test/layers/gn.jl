in_channel = 10
out_channel = 5
num_V = 6
num_E = 7
adj = [0. 1. 0. 0. 0. 0.;
       1. 0. 0. 1. 1. 1.;
       0. 0. 0. 0. 0. 1.;
       0. 1. 0. 0. 1. 0.;
       0. 1. 0. 1. 0. 1.;
       0. 1. 1. 0. 1. 0.]
ne = [[2], [1,4,5,6], [6], [2,5], [2,4,6], [2,3,5]]

struct NewGNLayer <: GraphNet
end

V = rand(in_channel, num_V)
E = rand(in_channel, 2*num_E)
u = rand(in_channel)

@testset "gn" begin
    l = NewGNLayer()

    @testset "without aggregation" begin
        (l::NewGNLayer)(fg) = propagate(l, fg)

        fg = FeaturedGraph(adj, V)
        fg_ = l(fg)

        @test graph(fg_) == adj
        @test size(node_feature(fg_)) == (in_channel, num_V)
        @test size(edge_feature(fg_)) == (0, 2*num_E)
        @test size(global_feature(fg_)) == (0,)
    end

    @testset "with neighbor aggregation" begin
        (l::NewGNLayer)(fg) = propagate(l, fg, :add)

        fg = FeaturedGraph(adj, V, E, zeros(0))
        l = NewGNLayer()
        fg_ = l(fg)

        @test graph(fg_) == adj
        @test size(node_feature(fg_)) == (in_channel, num_V)
        @test size(edge_feature(fg_)) == (in_channel, 2*num_E)
        @test size(global_feature(fg_)) == (0,)
    end

    GeometricFlux.update_edge(l::NewGNLayer, e, vi, vj, u) = rand(out_channel)
    @testset "update edge with neighbor aggregation" begin
        (l::NewGNLayer)(fg) = propagate(l, fg, :add)

        fg = FeaturedGraph(adj, V, E, zeros(0))
        l = NewGNLayer()
        fg_ = l(fg)

        @test graph(fg_) == adj
        @test size(node_feature(fg_)) == (in_channel, num_V)
        @test size(edge_feature(fg_)) == (out_channel, 2*num_E)
        @test size(global_feature(fg_)) == (0,)
    end

    GeometricFlux.update_vertex(l::NewGNLayer, ē, vi, u) = rand(out_channel)
    @testset "update edge/vertex with all aggregation" begin
        (l::NewGNLayer)(fg) = propagate(l, fg, :add, :add, :add)

        fg = FeaturedGraph(adj, V, E, u)
        l = NewGNLayer()
        fg_ = l(fg)

        @test graph(fg_) == adj
        @test size(node_feature(fg_)) == (out_channel, num_V)
        @test size(edge_feature(fg_)) == (out_channel, 2*num_E)
        @test size(global_feature(fg_)) == (in_channel,)
    end
end
