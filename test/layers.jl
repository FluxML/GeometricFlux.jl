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
    X = rand(N, in_channel)
    Y = gc(X)
    @test size(Y) == (N, out_channel)
end


@testset "Test ChebConv layer" begin
    k = 4
    cc = ChebConv(adj, in_channel=>out_channel, k)
    X = rand(N, in_channel)
    Y = cc(X)
    @test size(Y) == (N, out_channel)
end

@testset "Test GraphConv layer" begin
    gc = GraphConv(adj, in_channel=>out_channel)
    X = rand(N, in_channel)
    Y = gc(X)
    @test gc.edgelist == [[2,4], [1,3], [2,4], [1,3]]
    @test size(Y) == (N, out_channel)
end

@testset "Test GATConv layer" begin
    gat = GATConv(adj, in_channel=>out_channel)
    X = rand(N, in_channel)
    Y = gat(X)
    @test gat.edgelist == [[2,4], [1,3], [2,4], [1,3]]
    @test size(Y) == (N, out_channel)
end
