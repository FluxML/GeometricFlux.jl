in_channel = 3
out_channel = 5
N = 6
adj = [0. 1. 1. 0. 0. 0.;
       1. 0. 1. 0. 1. 0.;
       1. 1. 0. 1. 0. 1.;
       0. 0. 1. 0. 0. 0.;
       0. 1. 0. 0. 0. 0.;
       0. 0. 1. 0. 0. 0.]

sg = SimpleGraph(6)
add_edge!(sg, 1, 2); add_edge!(sg, 1, 3); add_edge!(sg, 2, 3)
add_edge!(sg, 3, 4); add_edge!(sg, 2, 5); add_edge!(sg, 3, 6)

ug = MetaGraph(sg)
dg = MetaDiGraph(sg)


@testset "metagraphs" begin
    @testset "GCNConv" begin
        gc = GCNConv(ug, in_channel=>out_channel)
        @test size(gc.weight) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel, N)
        @test size(gc.norm) == (N, N)

        gc = GCNConv(dg, in_channel=>out_channel)
        @test size(gc.weight) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel, N)
        @test size(gc.norm) == (N, N)
    end

    @testset "ChebConv" begin
        k = 4
        cc = ChebConv(ug, in_channel=>out_channel, k)
        @test size(cc.weight) == (out_channel, in_channel, k)
        @test size(cc.bias) == (out_channel, N)
        @test size(cc.L̃) == (N, N)
        @test cc.k == k
        @test cc.in_channel == in_channel
        @test cc.out_channel == out_channel

        cc = ChebConv(dg, in_channel=>out_channel, k)
        @test size(cc.weight) == (out_channel, in_channel, k)
        @test size(cc.bias) == (out_channel, N)
        @test size(cc.L̃) == (N, N)
        @test cc.k == k
        @test cc.in_channel == in_channel
        @test cc.out_channel == out_channel
    end

    @testset "GraphConv" begin
        gc = GraphConv(ug, in_channel=>out_channel)
        @test gc.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
        @test size(gc.weight1) == (out_channel, in_channel)
        @test size(gc.weight2) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel, N)

        gc = GraphConv(dg, in_channel=>out_channel)
        @test gc.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
        @test size(gc.weight1) == (out_channel, in_channel)
        @test size(gc.weight2) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel, N)
    end

    @testset "GATConv" begin
        gat = GATConv(ug, in_channel=>out_channel)
        @test gat.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
        @test size(gat.weight) == (out_channel, in_channel)
        @test size(gat.bias) == (out_channel, N)

        gat = GATConv(dg, in_channel=>out_channel)
        @test gat.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
        @test size(gat.weight) == (out_channel, in_channel)
        @test size(gat.bias) == (out_channel, N)
    end

    @testset "GatedGraphConv" begin
        num_layers = 3
        ggc = GatedGraphConv(ug, out_channel, num_layers)
        @test ggc.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
        @test size(ggc.weight) == (out_channel, out_channel, num_layers)

        ggc = GatedGraphConv(dg, out_channel, num_layers)
        @test ggc.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
        @test size(ggc.weight) == (out_channel, out_channel, num_layers)
    end

    @testset "EdgeConv" begin
        ec = EdgeConv(ug, Dense(2*in_channel, out_channel))
        @test ec.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]

        ec = EdgeConv(dg, Dense(2*in_channel, out_channel))
        @test ec.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
    end
end
