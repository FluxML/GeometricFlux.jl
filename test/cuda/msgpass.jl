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

struct NewCudaLayer <: MessagePassing
    weight
end
NewCudaLayer(m, n) = NewCudaLayer(randn(m,n))
@functor NewCudaLayer

(l::NewCudaLayer)(X) = propagate(l, X, :add)
GeometricFlux.message(n::NewCudaLayer, x_i, x_j, e_ij) = n.weight * x_j
GeometricFlux.update(::NewCudaLayer, m, x) = m

X = rand(Float32, in_channel, N) |> gpu
fg = FeaturedGraph(adj, X)
l = NewCudaLayer(out_channel, in_channel) |> gpu

@testset "cuda/msgpass" begin
    fg_ = l(fg)
    @test size(node_feature(fg_)) == (out_channel, N)
end
