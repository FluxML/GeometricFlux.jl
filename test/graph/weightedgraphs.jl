using SimpleWeightedGraphs
using LightGraphs: add_edge!

in_channel = 3
out_channel = 5
N = 6
adj = [0. 1. 1. 0. 0. 0.;
       1. 0. 1. 0. 1. 0.;
       1. 1. 0. 1. 0. 1.;
       0. 0. 1. 0. 0. 0.;
       0. 1. 0. 0. 0. 0.;
       0. 0. 1. 0. 0. 0.]

ug = SimpleWeightedGraph(6)
add_edge!(ug, 1, 2, 2); add_edge!(ug, 1, 3, 2); add_edge!(ug, 2, 3, 1)
add_edge!(ug, 3, 4, 5); add_edge!(ug, 2, 5, 2); add_edge!(ug, 3, 6, 2)

@testset "Test support of SimpleWeightedGraphs for GCNConv layer" begin
    gc = GCNConv(ug, in_channel=>out_channel)
    @test size(gc.weight) == (out_channel, in_channel)
    @test size(gc.bias) == (out_channel, N)
    @test size(gc.norm) == (N, N)
end


@testset "Test support of SimpleWeightedGraphs for ChebConv layer" begin
    k = 4
    cc = ChebConv(ug, in_channel=>out_channel, k)
    @test size(cc.weight) == (out_channel, in_channel, k)
    @test size(cc.bias) == (out_channel, N)
    @test size(cc.L̃) == (N, N)
    @test cc.k == k
    @test cc.in_channel == in_channel
    @test cc.out_channel == out_channel
end

@testset "Test support of SimpleWeightedGraphs for GraphConv layer" begin
    gc = GraphConv(ug, in_channel=>out_channel)
    @test gc.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
    @test size(gc.weight1) == (out_channel, in_channel)
    @test size(gc.weight2) == (out_channel, in_channel)
    @test size(gc.bias) == (out_channel, N)
end

@testset "Test support of SimpleWeightedGraphs for GATConv layer" begin
    gat = GATConv(ug, in_channel=>out_channel)
    @test gat.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
    @test size(gat.weight) == (out_channel, in_channel)
    @test size(gat.bias) == (out_channel, N)
end

@testset "Test support of SimpleWeightedGraphs for GatedGraphConv layer" begin
    num_layers = 3
    ggc = GatedGraphConv(ug, out_channel, num_layers)
    @test ggc.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
    @test size(ggc.weight) == (out_channel, out_channel, num_layers)
end

@testset "Test support of SimpleWeightedGraphs for EdgeConv layer" begin
    ec = EdgeConv(ug, Dense(2*in_channel, out_channel))
    @test ec.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
end
