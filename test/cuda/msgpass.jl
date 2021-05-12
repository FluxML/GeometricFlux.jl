in_channel = 10
out_channel = 5
N = 6
T = Float32
adj = [0. 1. 0. 0. 0. 0.;
       1. 0. 0. 1. 1. 1.;
       0. 0. 0. 0. 0. 1.;
       0. 1. 0. 0. 1. 0.;
       0. 1. 0. 1. 0. 1.;
       0. 1. 1. 0. 1. 0.]

struct NewCudaLayer <: MessagePassing
    weight
end
NewCudaLayer(m, n) = NewCudaLayer(randn(T, m,n))
@functor NewCudaLayer

(l::NewCudaLayer)(X) = propagate(l, X, +)
GeometricFlux.message(n::NewCudaLayer, x_i, x_j, e_ij) = n.weight * x_j
GeometricFlux.update(::NewCudaLayer, m, x) = m

X = rand(T, in_channel, N) |> gpu
fg = FeaturedGraph(adj, nf=X)
l = NewCudaLayer(out_channel, in_channel) |> gpu

@testset "cuda/msgpass" begin
    fg_ = l(fg)
    @test size(node_feature(fg_)) == (out_channel, N)
end
