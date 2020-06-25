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

(l::NewLayer)(X) = propagate(l, :add, X=X)
message(::NewLayer, x_j) = x_j
update(::NewLayer, M) = M

X = Array(reshape(1:N*in_channel, in_channel, N))
l = NewLayer(adj, out_channel, in_channel)

message(n::NewLayer, x_j) = n.weight * x_j

@testset "msgpass" begin
    Y = l(X)
    @test size(Y) == (out_channel, N)
end
