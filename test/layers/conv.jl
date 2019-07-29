using Flux: Dense
import GeometricFlux: message, update, propagate

in_channel = 3
out_channel = 5
N = 4
adj = [0. 1. 0. 1.;
       1. 0. 1. 0.;
       0. 1. 0. 1.;
       1. 0. 1. 0.]


struct NewLayer <: MessagePassing
    adjlist
    weight
    NewLayer(adj, m, n) = new(neighbors(adj), randn(m,n))
end

(l::NewLayer)(X) = propagate(l, X=X, aggr=:add)
message(n::NewLayer; X::AbstractArray=zeros(0), E::AbstractArray=zeros(0)) = X * n.weight
update(::NewLayer; X::AbstractArray=zeros(0), M::AbstractArray=zeros(0)) = M

@testset "Test MessagePassing layer" begin
    l = NewLayer(adj, in_channel, out_channel)
    X = rand(N, in_channel)
    Y = l(X)
    @test size(Y) == (N, out_channel)
end


@testset "Test GCNConv layer" begin
    gc = GCNConv(adj, in_channel=>out_channel)
    @test size(gc.weight) == (in_channel, out_channel)
    @test size(gc.bias) == (N, out_channel)
    @test size(gc.norm) == (N, N)

    X = rand(N, in_channel)
    Y = gc(X)
    @test size(Y) == (N, out_channel)
end


@testset "Test ChebConv layer" begin
    k = 4
    cc = ChebConv(adj, in_channel=>out_channel, k)
    @test size(cc.weight) == (k, in_channel, out_channel)
    @test size(cc.bias) == (N, out_channel)
    @test size(cc.L̃) == (N, N)
    @test cc.k == k
    @test cc.in_channel == in_channel
    @test cc.out_channel == out_channel

    X = rand(N, in_channel)
    Y = cc(X)
    @test size(Y) == (N, out_channel)
end

@testset "Test GraphConv layer" begin
    gc = GraphConv(adj, in_channel=>out_channel)
    @test gc.adjlist == [[2,4], [1,3], [2,4], [1,3]]
    @test size(gc.weight1) == (in_channel, out_channel)
    @test size(gc.weight2) == (in_channel, out_channel)
    @test size(gc.bias) == (N, out_channel)

    X = rand(N, in_channel)
    Y = gc(X)
    @test size(Y) == (N, out_channel)
end

@testset "Test GATConv layer" begin
    gat = GATConv(adj, in_channel=>out_channel)
    @test gat.adjlist == [[2,4], [1,3], [2,4], [1,3]]
    @test size(gat.weight) == (in_channel, out_channel)
    @test size(gat.bias) == (N, out_channel)

    X = rand(N, in_channel)
    Y = gat(X)
    @test size(Y) == (N, out_channel)
end

@testset "Test EdgeConv layer" begin
    ec = EdgeConv(adj, Dense(2*in_channel, out_channel))
    @test ec.adjlist == [[2,4], [1,3], [2,4], [1,3]]

    X = rand(N, in_channel)
    Y = ec(X)
    @test size(Y) == (N, out_channel)
end
