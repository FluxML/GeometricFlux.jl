in_channel = 3
out_channel = 5
N = 6
adj = [0. 2. 2. 0. 0. 0.;
       2. 0. 1. 0. 2. 0.;
       2. 1. 0. 5. 0. 2.;
       0. 0. 5. 0. 0. 0.;
       0. 2. 0. 0. 0. 0.;
       0. 0. 2. 0. 0. 0.]
deg = [4. 0. 0. 0. 0. 0.;
       0. 5. 0. 0. 0. 0.;
       0. 0. 10. 0. 0. 0.;
       0. 0. 0. 5. 0. 0.;
       0. 0. 0. 0. 2. 0.;
       0. 0. 0. 0. 0. 2.]
lap = [4. -2. -2. 0. 0. 0.;
       -2. 5. -1. 0. -2. 0.;
       -2. -1. 10. -5. 0. -2.;
       0. 0. -5. 5. 0. 0.;
       0. -2. 0. 0. 2. 0.;
       0. 0. -2. 0. 0. 2.]
norm_lap = [1. -2/sqrt(4*5) -2/sqrt(4*10) 0. 0. 0.;
            -2/sqrt(4*5) 1. -1/sqrt(5*10) 0. -2/sqrt(2*5) 0.;
            -2/sqrt(4*10) -1/sqrt(5*10) 1. -5/sqrt(5*10) 0. -2/sqrt(2*10);
            0. 0. -5/sqrt(5*10) 1. 0. 0.;
            0. -2/sqrt(2*5) 0. 0. 1. 0.;
            0. 0. -2/sqrt(2*10) 0. 0. 1.]

ug = SimpleWeightedGraph(6)
add_edge!(ug, 1, 2, 2); add_edge!(ug, 1, 3, 2); add_edge!(ug, 2, 3, 1)
add_edge!(ug, 3, 4, 5); add_edge!(ug, 2, 5, 2); add_edge!(ug, 3, 6, 2)


@testset "weightedgraphs" begin
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
        @test graph(gc.graph)[] === ug
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
    end

    @testset "GraphConv" begin
        gc = GraphConv(ug, in_channel=>out_channel)
        @test gc.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
        @test size(gc.weight1) == (out_channel, in_channel)
        @test size(gc.weight2) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)
    end

    @testset "GATConv" begin
        for heads = [1, 5]
            for concat = [true, false]
                gat = GATConv(ug, in_channel=>out_channel, heads=heads, concat=concat)
                @test gat.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
                @test size(gat.weight) == (out_channel * heads, in_channel)
                @test size(gat.bias) == (out_channel * heads,)
                @test size(gat.a) == (2*out_channel, heads, 1)
            end
        end
    end

    @testset "GatedGraphConv" begin
        num_layers = 3
        ggc = GatedGraphConv(ug, out_channel, num_layers)
        @test ggc.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
        @test size(ggc.weight) == (out_channel, out_channel, num_layers)
    end

    @testset "EdgeConv" begin
        ec = EdgeConv(ug, Dense(2*in_channel, out_channel))
        @test ec.adjlist == [[2,3], [1,3,5], [1,2,4,6], [3], [2], [3]]
    end
end
