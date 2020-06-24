in_channel = 3
out_channel = 5
N = 6
adj = [0. 1. 1. 0. 0. 0.;
       1. 0. 1. 0. 1. 0.;
       1. 1. 0. 1. 0. 1.;
       0. 0. 1. 0. 0. 0.;
       0. 1. 0. 0. 0. 0.;
       0. 0. 1. 0. 0. 0.]
deg = [2. 0. 0. 0. 0. 0.;
       0. 3. 0. 0. 0. 0.;
       0. 0. 4. 0. 0. 0.;
       0. 0. 0. 1. 0. 0.;
       0. 0. 0. 0. 1. 0.;
       0. 0. 0. 0. 0. 1.]
lap = [2. -1. -1. 0. 0. 0.;
       -1. 3. -1. 0. -1. 0.;
       -1. -1. 4. -1. 0. -1.;
       0. 0. -1. 1. 0. 0.;
       0. -1. 0. 0. 1. 0.;
       0. 0. -1. 0. 0. 1.]
norm_lap = [1. -1/sqrt(2*3) -1/sqrt(2*4) 0. 0. 0.;
            -1/sqrt(2*3) 1. -1/sqrt(3*4) 0. -1/sqrt(3) 0.;
            -1/sqrt(2*4) -1/sqrt(3*4) 1. -1/2 0. -1/2;
            0. 0. -1/2 1. 0. 0.;
            0. -1/sqrt(3) 0. 0. 1. 0.;
            0. 0. -1/2 0. 0. 1.]

sg = SimpleGraph(6)
add_edge!(sg, 1, 2); add_edge!(sg, 1, 3); add_edge!(sg, 2, 3)
add_edge!(sg, 3, 4); add_edge!(sg, 2, 5); add_edge!(sg, 3, 6)

ug = MetaGraph(sg)
dg = MetaDiGraph(sg)


@testset "metagraphs" begin
    @testset "linalg" begin
        for T in [Int8, Int16, Int32, Int64, Int128]
            @test degree_matrix(adj, T, dir=:out) == T.(deg)
            @test degree_matrix(adj, T, dir=:out) == degree_matrix(adj, T, dir=:in)
            @test degree_matrix(adj, T, dir=:out) == degree_matrix(adj, T, dir=:both)
            @test laplacian_matrix(adj, T) == T.(lap)
        end
        for T in [Float16, Float32, Float64]
            @test degree_matrix(adj, T, dir=:out) == T.(deg)
            @test degree_matrix(adj, T, dir=:out) == degree_matrix(adj, T, dir=:in)
            @test degree_matrix(adj, T, dir=:out) == degree_matrix(adj, T, dir=:both)
            @test laplacian_matrix(adj, T) == T.(lap)
            @test normalized_laplacian(adj, T) ≈ T.(norm_lap)
        end
    end

    @testset "GCNConv" begin
        gc = GCNConv(ug, in_channel=>out_channel)
        @test size(gc.weight) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)
        @test graph(gc.graph) === ug.graph

        gc = GCNConv(dg, in_channel=>out_channel)
        @test size(gc.weight) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)
        @test graph(gc.graph) === dg.graph
    end

    @testset "ChebConv" begin
        k = 4
        cc = ChebConv(ug, in_channel=>out_channel, k)
        @test size(cc.weight) == (out_channel, in_channel, k)
        @test size(cc.bias) == (out_channel,)
        @test size(cc.L̃) == (N, N)
        @test cc.k == k
        @test cc.in_channel == in_channel
        @test cc.out_channel == out_channel

        cc = ChebConv(dg, in_channel=>out_channel, k)
        @test size(cc.weight) == (out_channel, in_channel, k)
        @test size(cc.bias) == (out_channel,)
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
        @test size(gc.bias) == (out_channel,)

        gc = GraphConv(dg, in_channel=>out_channel)
        @test gc.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
        @test size(gc.weight1) == (out_channel, in_channel)
        @test size(gc.weight2) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)
    end

    @testset "GATConv" begin
        gat = GATConv(ug, in_channel=>out_channel)
        @test gat.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
        @test size(gat.weight) == (out_channel, in_channel)
        @test size(gat.bias) == (out_channel,)

        gat = GATConv(dg, in_channel=>out_channel)
        @test gat.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
        @test size(gat.weight) == (out_channel, in_channel)
        @test size(gat.bias) == (out_channel,)
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
