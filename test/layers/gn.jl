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

struct NewGNLayer <: GraphNet
end

(l::NewGNLayer)(fg) = propagate(l, fg)

X = Array(reshape(1:num_V*in_channel, in_channel, num_V))

@testset "gn" begin
    fg = FeaturedGraph(adj, X)
    l = NewGNLayer()
    fg_ = l(fg)

    @test graph(fg_) === adj
    @test size(node_feature(fg_)) == (in_channel, num_V)
    @test size(edge_feature(fg_)) == (0, 2*num_E)
    @test size(global_feature(fg_)) == (0,)
end
