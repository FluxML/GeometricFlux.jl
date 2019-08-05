import GeometricFlux: message, update, propagate

in_channel = 10
out_channel = 5
N = 6
adj = [0. 1. 0. 0. 0. 0.;
       1. 0. 0. 1. 1. 1.;
       0. 0. 0. 0. 0. 1.;
       0. 1. 0. 0. 1. 0.;
       0. 1. 0. 1. 0. 1.;
       0. 1. 1. 0. 1. 0.]
ne = [[2], [1,4,5,6], [6], [2,5], [2,4,6], [2,3,5]]

struct NewLayer <: MessagePassing
    adjlist::AbstractVector{<:AbstractVector}
    weight
end

NewLayer(adjm::AbstractMatrix, m, n) = NewLayer(neighbors(adjm), randn(m,n))

(l::NewLayer)(X) = propagate(l, X=X, aggr=:add)
message(n::NewLayer; x_i=zeros(0), x_j=zeros(0)) = x_j
update(::NewLayer; X=zeros(0), M=zeros(0)) = M

X = Array(reshape(1:N*in_channel, N, in_channel))
l = NewLayer(adj, in_channel, out_channel)

@testset "Test neighboring" begin
    Y, clst = neighboring(l; X=X)
    @test adjlist(l) == ne
    @test size(Y) == (in_channel, 14)
    @test Y[1, :] == [2, 1, 4, 5, 6, 6, 2, 5, 2, 4, 6, 2, 3, 5]
    @test clst == [1, 2, 2, 2, 2, 3, 4, 4, 5, 5, 5, 6 ,6, 6]
end

message(n::NewLayer; x_i=zeros(0), x_j=zeros(0)) = n.weight' * x_j

@testset "Test MessagePassing layer" begin
    Y = l(X)
    @test size(Y) == (N, out_channel)
end

in_channel = 100
X = Array(reshape(1:N*in_channel, N, in_channel))
l = NewLayer(adj, in_channel, out_channel)

@testset "Test multi-thread MessagePassing layer" begin
    Y = l(X)
    @test size(Y) == (N, out_channel)
end
