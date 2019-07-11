using Flux: Dense
import GeometricFlux: message, update, propagate

const in_channel = 3
const out_channel = 5
const N = 4
const adj = [0. 1. 0. 1.;
             1. 0. 1. 0.;
             0. 1. 0. 1.;
             1. 0. 1. 0.]


struct NewLayer <: MessagePassing
    weight
    NewLayer(m, n) = new(randn(m,n))
end

(l::NewLayer)(X) = propagate(l, neighbors(adj), X, zeros(10, N, N), aggr=:+)
message(n::NewLayer, xi, xj, eij) = n.weight' * xj
update(::NewLayer, xi, mi) = mi

@testset "Test MessagePassing layer" begin
    l = NewLayer(in_channel, out_channel)
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
    @test gc.edgelist == [[2,4], [1,3], [2,4], [1,3]]
    @test size(gc.weight) == (in_channel, out_channel)
    @test size(gc.bias) == (N, out_channel)

    X = rand(N, in_channel)
    Y = gc(X)
    @test size(Y) == (N, out_channel)
end

@testset "Test GATConv layer" begin
    gat = GATConv(adj, in_channel=>out_channel)
    @test gat.edgelist == [[2,4], [1,3], [2,4], [1,3]]
    @test size(gat.weight) == (in_channel, out_channel)
    @test size(gat.bias) == (N, out_channel)

    X = rand(N, in_channel)
    Y = gat(X)
    @test size(Y) == (N, out_channel)
end

@testset "Test EdgeConv layer" begin
    ec = EdgeConv(adj, Dense(2*in_channel, out_channel))
    @test ec.edgelist == [[2,4], [1,3], [2,4], [1,3]]

    X = rand(N, in_channel)
    Y = ec(X)
    @test size(Y) == (N, out_channel)
end
