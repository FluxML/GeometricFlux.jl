using SimpleWeightedGraphs
using LightGraphs: add_edge!

const in_channel = 3
const out_channel = 5
const N = 6
const adj = [0. 1. 1. 0. 0. 0.;
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
    @test size(gc.weight) == (in_channel, out_channel)
    @test size(gc.bias) == (N, out_channel)
    @test size(gc.norm) == (N, N)
end


@testset "Test support of SimpleWeightedGraphs for ChebConv layer" begin
    k = 4
    cc = ChebConv(ug, in_channel=>out_channel, k)
    @test size(cc.weight) == (k, in_channel, out_channel)
    @test size(cc.bias) == (N, out_channel)
    @test size(cc.L̃) == (N, N)
    @test cc.k == k
    @test cc.in_channel == in_channel
    @test cc.out_channel == out_channel
end

@testset "Test support of SimpleWeightedGraphs for GraphConv layer" begin
    gc = GraphConv(ug, in_channel=>out_channel)
    @test gc.edgelist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
    @test size(gc.weight) == (in_channel, out_channel)
    @test size(gc.bias) == (N, out_channel)
end

@testset "Test support of SimpleWeightedGraphs for GATConv layer" begin
    gat = GATConv(ug, in_channel=>out_channel)
    @test gat.edgelist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
    @test size(gat.weight) == (in_channel, out_channel)
    @test size(gat.bias) == (N, out_channel)
end

@testset "Test support of SimpleWeightedGraphs for EdgeConv layer" begin
    ec = EdgeConv(ug, Dense(2*in_channel, out_channel))
    @test ec.edgelist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
end
