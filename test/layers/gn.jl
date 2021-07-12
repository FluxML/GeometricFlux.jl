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

struct NewGNLayer <: GraphNet
end

V = rand(T, in_channel, num_V)
E = rand(T, in_channel, 2num_E)
u = rand(T, in_channel)

@testset "gn" begin
    l = NewGNLayer()

    @testset "without aggregation" begin
        (l::NewGNLayer)(fg) = GeometricFlux.propagate(l, fg)

        fg = FeaturedGraph(adj, nf=V)
        fg_ = l(fg)

        @test graph(fg_) === adj
        @test size(node_feature(fg_)) == (in_channel, num_V)
        @test size(edge_feature(fg_)) == (0, 2*num_E)
        @test size(global_feature(fg_)) == (0,)
    end

    @testset "with neighbor aggregation" begin
        (l::NewGNLayer)(fg) = GeometricFlux.propagate(l, fg, +)

        fg = FeaturedGraph(adj, nf=V, ef=E, gf=zeros(0))
        l = NewGNLayer()
        fg_ = l(fg)

        @test graph(fg_) === adj
        @test size(node_feature(fg_)) == (in_channel, num_V)
        @test size(edge_feature(fg_)) == (in_channel, 2*num_E)
        @test size(global_feature(fg_)) == (0,)
    end

    GeometricFlux.update_edge(l::NewGNLayer, e, vi, vj, u) = rand(T, out_channel)
    @testset "update edge with neighbor aggregation" begin
        (l::NewGNLayer)(fg) = GeometricFlux.propagate(l, fg, +)

        fg = FeaturedGraph(adj, nf=V, ef=E, gf=zeros(0))
        l = NewGNLayer()
        fg_ = l(fg)

        @test graph(fg_) === adj
        @test size(node_feature(fg_)) == (in_channel, num_V)
        @test size(edge_feature(fg_)) == (out_channel, 2*num_E)
        @test size(global_feature(fg_)) == (0,)
    end

    GeometricFlux.update_vertex(l::NewGNLayer, ē, vi, u) = rand(T, out_channel)
    @testset "update edge/vertex with all aggregation" begin
        (l::NewGNLayer)(fg) = GeometricFlux.propagate(l, fg, +, +, +)

        fg = FeaturedGraph(adj, nf=V, ef=E, gf=u)
        l = NewGNLayer()
        fg_ = l(fg)

        @test graph(fg_) === adj
        @test size(node_feature(fg_)) == (out_channel, num_V)
        @test size(edge_feature(fg_)) == (out_channel, 2*num_E)
        @test size(global_feature(fg_)) == (in_channel,)
    end
end
