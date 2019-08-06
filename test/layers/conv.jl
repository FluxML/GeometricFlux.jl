using Flux: Dense

in_channel = 3
out_channel = 5
N = 4
adj = [0. 1. 0. 1.;
       1. 0. 1. 0.;
       0. 1. 0. 1.;
       1. 0. 1. 0.]


@testset "Test GCNConv layer" begin
    gc = GCNConv(adj, in_channel=>out_channel)
    @test size(gc.weight) == (out_channel, in_channel)
    @test size(gc.bias) == (out_channel, N)
    @test size(gc.norm) == (N, N)

    X = rand(in_channel, N)
    Y = gc(X)
    @test size(Y) == (out_channel, N)
end


@testset "Test ChebConv layer" begin
    k = 6
    cc = ChebConv(adj, in_channel=>out_channel, k)
    @test size(cc.weight) == (out_channel, in_channel, k)
    @test size(cc.bias) == (out_channel, N)
    @test size(cc.L̃) == (N, N)
    @test cc.k == k
    @test cc.in_channel == in_channel
    @test cc.out_channel == out_channel

    X = rand(in_channel, N)
    Y = cc(X)
    @test size(Y) == (out_channel, N)
end

@testset "Test GraphConv layer" begin
    gc = GraphConv(adj, in_channel=>out_channel)
    @test gc.adjlist == [[2,4], [1,3], [2,4], [1,3]]
    @test size(gc.weight1) == (out_channel, in_channel)
    @test size(gc.weight2) == (out_channel, in_channel)
    @test size(gc.bias) == (out_channel, N)

    X = rand(in_channel, N)
    Y = gc(X)
    @test size(Y) == (out_channel, N)
end

@testset "Test GATConv layer" begin
    gat = GATConv(adj, in_channel=>out_channel)
    @test gat.adjlist == [[2,4], [1,3], [2,4], [1,3]]
    @test size(gat.weight) == (out_channel, in_channel)
    @test size(gat.bias) == (out_channel, N)

    X = rand(in_channel, N)
    Y = gat(X)
    @test size(Y) == (out_channel, N)
end

@testset "Test GatedGraphConv layer" begin
    num_layers = 3
    ggc = GatedGraphConv(adj, out_channel, num_layers)
    @test ggc.adjlist == [[2,4], [1,3], [2,4], [1,3]]
    @test size(ggc.weight) == (out_channel, out_channel, num_layers)

    X = rand(in_channel, N)
    Y = ggc(X)
    @test size(Y) == (out_channel, N)
end

@testset "Test EdgeConv layer" begin
    ec = EdgeConv(adj, Dense(2*in_channel, out_channel))
    @test ec.adjlist == [[2,4], [1,3], [2,4], [1,3]]

    X = rand(in_channel, N)
    Y = ec(X)
    @test size(Y) == (out_channel, N)
end
