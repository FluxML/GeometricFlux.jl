using GeometricFlux: update_edge, update_vertex, update_global

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

V = rand(in_channel, num_V)
E = rand(in_channel, 2*num_E)
u = rand(in_channel)

@testset "gn" begin
    # Without aggregation
    struct NewGNLayer <: GraphNet
    end

    (l::NewGNLayer)(fg) = propagate(l, fg)

    fg = FeaturedGraph(adj, V)
    l = NewGNLayer()
    fg_ = l(fg)

    @test graph(fg_) === adj
    @test size(node_feature(fg_)) == (in_channel, num_V)
    @test size(edge_feature(fg_)) == (0, 2*num_E)
    @test size(global_feature(fg_)) == (0,)

    # With neighbor aggregation
    struct NewGNLayer2 <: GraphNet
    end

    (l::NewGNLayer2)(fg) = propagate(l, fg, :add)

    fg = FeaturedGraph(adj, V, E, zeros(0))
    l = NewGNLayer2()
    fg_ = l(fg)

    @test graph(fg_) === adj
    @test size(node_feature(fg_)) == (in_channel, num_V)
    @test size(edge_feature(fg_)) == (in_channel, 2*num_E)
    @test size(global_feature(fg_)) == (0,)


    # With neighbor and global aggregation
    struct NewGNLayer3 <: GraphNet
    end

    (l::NewGNLayer3)(fg) = propagate(l, fg, :add, :add, :add)

    fg = FeaturedGraph(adj, V, E, u)
    l = NewGNLayer3()
    fg_ = l(fg)

    @test graph(fg_) === adj
    @test size(node_feature(fg_)) == (in_channel, num_V)
    @test size(edge_feature(fg_)) == (in_channel, 2*num_E)
    @test size(global_feature(fg_)) == (in_channel,)
end
