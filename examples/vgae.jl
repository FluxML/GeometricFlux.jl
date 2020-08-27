using GeometricFlux
using Flux
using Flux: throttle
using Flux.Losses: logitbinarycrossentropy
using Flux: @epochs
using JLD2  # use v0.1.2
using Distributions
using SparseArrays
using LightGraphs.SimpleGraphs
using LightGraphs: adjacency_matrix

@load "data/cora_features.jld2" features
@load "data/cora_graph.jld2" g

num_nodes = 2708
num_features = 1433
hidden1 = 128
hidden2 = 32
z_dim = 16
target_catg = 7
epochs = 200

## Preprocessing data
adj_mat = Matrix{Float32}(adjacency_matrix(g))
train_data = [(Matrix{Float32}(features), adj_mat)]

## Model
encoder = Chain(GCNConv(adj_mat, num_features=>hidden1, relu),
                GCNConv(adj_mat, hidden1=>hidden2))
model = VGAE(encoder, hidden2, z_dim)
encoder = model.encoder
decoder = model.decoder
ps = Flux.params(model)

l2_norm(p) = sum(abs2, p)

function loss(X, Y, T=eltype(X), β=one(T), λ=T(0.01); debug=false)
    μ̂, logσ̂ = summarize(encoder, X)
    Z = encoder(X)
    kl_q_p = -T(0.5) * sum(one(T) .+ T(2).*logσ̂ .- μ̂.^2 .- exp.(T(2).*logσ̂))
    logp_y_z = -sum(logitbinarycrossentropy(decoder(Z), Y, agg=identity)) / size(Y,2)
    l2reg = sum(l2_norm, ps)
    debug && begin
        @show kl_q_p
        @show logp_y_z
        @show l2reg
    end
    -logp_y_z + β*kl_q_p + λ*l2reg
end

## Training
opt = ADAM(0.001)
evalcb() = @show(loss(train_data[1]...))

@epochs epochs Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
