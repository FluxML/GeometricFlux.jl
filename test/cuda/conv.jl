using Flux: Dense, gpu

in_channel = 3
out_channel = 5
N = 4
adj = [0 1 0 1;
       1 0 1 0;
       0 1 0 1;
       1 0 1 0]


@testset "cuda/conv" begin
    @testset "GCNConv" begin
        gc = GCNConv(adj, in_channel=>out_channel) |> gpu
        @test size(gc.weight) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel, N)
        @test size(gc.norm) == (N, N)

        X = rand(in_channel, N) |> gpu
        Y = gc(X)
        @test size(Y) == (out_channel, N)
    end


    @testset "ChebConv" begin
        k = 6
        cc = ChebConv(adj, in_channel=>out_channel, k) |> gpu
        @test size(cc.weight) == (out_channel, in_channel, k)
        @test size(cc.bias) == (out_channel, N)
        @test size(cc.L̃) == (N, N)
        @test cc.k == k
        @test cc.in_channel == in_channel
        @test cc.out_channel == out_channel

        X = rand(in_channel, N) |> gpu
        Y = cc(X)
        @test size(Y) == (out_channel, N)
    end

    @testset "GraphConv" begin
        gc = GraphConv(adj, in_channel=>out_channel) |> gpu
        @test gc.adjlist == [[2,4], [1,3], [2,4], [1,3]]
        @test eltype(gc.adjlist[1]) <: Integer
        @test size(gc.weight1) == (out_channel, in_channel)
        @test size(gc.weight2) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel, N)

        X = rand(in_channel, N) |> gpu
        Y = gc(X)
        @test size(Y) == (out_channel, N)
    end

    @testset "GATConv" begin
        gat = GATConv(adj, in_channel=>out_channel) |> gpu
        @test gat.adjlist == [[2,4], [1,3], [2,4], [1,3]]
        @test eltype(gat.adjlist[1]) <: Integer
        @test size(gat.weight) == (out_channel, in_channel)
        @test size(gat.bias) == (out_channel, N)

        X = rand(in_channel, N) |> gpu
        Y = gat(X)
        @test size(Y) == (out_channel, N)
    end

    @testset "GatedGraphConv" begin
        num_layers = 3
        ggc = GatedGraphConv(adj, out_channel, num_layers) |> gpu
        @test ggc.adjlist == [[2,4], [1,3], [2,4], [1,3]]
        @test eltype(ggc.adjlist[1]) <: Integer
        @test size(ggc.weight) == (out_channel, out_channel, num_layers)

        X = rand(in_channel, N) |> gpu
        Y = ggc(X)
        @test size(Y) == (out_channel, N)
    end

    @testset "EdgeConv" begin
        ec = EdgeConv(adj, Dense(2*in_channel, out_channel)) |> gpu
        @test ec.adjlist == [[2,4], [1,3], [2,4], [1,3]]
        @test eltype(ec.adjlist[1]) <: Integer

        X = rand(in_channel, N) |> gpu
        Y = ec(X)
        @test size(Y) == (out_channel, N)
    end
end
