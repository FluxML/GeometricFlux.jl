using GeometricFlux
using Flux
using Flux: throttle
using Flux: @epochs
using JLD2  # use v0.1.2
using Distributions
using SparseArrays
using LightGraphs.SimpleGraphs
using LightGraphs: adjacency_matrix
using CUDA

import Distributions: logpdf

@load "data/cora_features.jld2" features
@load "data/cora_graph.jld2" g

num_nodes = 2708
num_features = 1433
h_dim = 32
z_dim = 16
target_catg = 7
epochs = 200

## Preprocessing data
adj_mat = Matrix{Float32}(adjacency_matrix(g))# |> gpu
train_X = [Matrix{Float32}(features) for i = 1:10]# |> gpu  # dim: num_features * num_nodes
train_y = [adj_mat for i = 1:10]  # dim: num_nodes * num_nodes

## Model
model = VGAE(GCNConv(adj_mat, num_features=>h_dim, relu),
             h_dim, z_dim)# |> gpu
encoder = model.encoder
decoder = model.decoder

logpdf(b::Bernoulli, y::Bool, T::Type{Real}=Float32) =
    y * log(b.p + eps(T)) + (one(T) - y) * log(one(T) - b.p + eps(T))

# KL-divergence between approximation posterior and N(0, 1) prior.
kl_q_p(μ, logσ, T::Type{Real}=Float32) =
    T(0.5) * sum(exp.(T(2) .* logσ) + μ.^2 .- one(T) .+ logσ.^2)

# logp(x|z) - decoder
logp_x_z(x, z) = sum(logpdf.(Bernoulli.(decoder(z)), x))

function loss(X, Y, T=eltype(X), β=one(T), λ=T(0.01))
    N = length(X)
    f = (x) -> summarize(encoder, encoder.nn(x))
    stats = f.(X)
    μ̂ = [s[1] for s in stats]
    logσ̂ = [s[2] for s in stats]
    Z = encoder.(X)
    L̄ = 1//N * (logp_x_z(X, Z) - β * kl_q_p(μ̂, logσ̂))
    -L̄ + λ * sum(abs2, Flux.params(model))
end

## Training
ps = Flux.params(model)
train_data = [(train_X, train_y)]
opt = ADAM(0.01)
evalcb() = @show(loss(train_X, train_y))

@epochs epochs Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
