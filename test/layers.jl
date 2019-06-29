const in_channel = 3
const out_channel = 5
const N = 4
const adj = [0. 1. 0. 1.;
             1. 0. 1. 0.;
             0. 1. 0. 1.;
             1. 0. 1. 0.]

# @testset "Test MessagePassing layer" begin
#     struct NewLayer <: MessagePassing
#         weight
#         NewLayer(m, n) = new(randn(m,n))
#     end
#     message(n::NewLayer, xi, xj, eij) = n.weight * xj
#     update(::NewLayer, xi, mi) = mi
#
#     l = NewLayer(4, 5)
#     X = rand()
#     propagate(l, neighbors, X, E, aggr=+)
# end


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
