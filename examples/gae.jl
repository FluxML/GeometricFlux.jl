using GeometricFlux
using Flux
using Flux: throttle
using Flux.Losses: logitbinarycrossentropy
using Flux: @epochs
using JLD2
using Statistics: mean
using SparseArrays
using Graphs.SimpleGraphs
using Graphs: adjacency_matrix
using CUDA

@load "data/cora_features.jld2" features
@load "data/cora_graph.jld2" g

num_nodes = 2708
num_features = 1433
hidden1 = 32
hidden2 = 16
target_catg = 7
epochs = 200

## Preprocessing data
adj_mat = Matrix{Float32}(adjacency_matrix(g)) |> gpu
train_X = Matrix{Float32}(features) |> gpu  # dim: num_features * num_nodes
train_y = adj_mat  # dim: num_nodes * num_nodes

## Model
encoder = Chain(GCNConv(adj_mat, num_features=>hidden1, relu),
                GCNConv(adj_mat, hidden1=>hidden2))
model = Chain(GAE(encoder, σ)) |> gpu

## Loss
loss(x, y) = logitbinarycrossentropy(model(x), y)

## Training
ps = Flux.params(model)
train_data = [(train_X, train_y)]
opt = ADAM(0.01)
evalcb() = @show(loss(train_X, train_y))

@epochs epochs Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
