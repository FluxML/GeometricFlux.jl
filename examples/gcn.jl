using GeometricFlux
using Flux
using Flux: onehotbatch, onecold, logitcrossentropy, throttle
using Flux: @epochs
using JLD2
using Statistics
using SparseArrays
using LightGraphs.SimpleGraphs
using LightGraphs: adjacency_matrix
using CUDA
using Random

Random.seed!([0x6044b4da, 0xd873e4f9, 0x59d90c0a, 0xde01aa81])

@load "data/cora_features.jld2" features
@load "data/cora_labels.jld2" labels
@load "data/cora_graph.jld2" g

num_nodes = 2708
num_features = 1433
hidden = 16
target_catg = 7
epochs = 100

## Preprocessing data
train_X = Matrix{Float32}(features) |> gpu  # dim: num_features * num_nodes
train_y = Matrix{Float32}(labels) |> gpu  # dim: target_catg * num_nodes
adj_mat = Matrix{Float32}(adjacency_matrix(g)) |> gpu

## Model
model = Chain(GCNConv(adj_mat, num_features=>hidden, relu),
              Dropout(0.5),
              GCNConv(adj_mat, hidden=>target_catg),
              softmax) |> gpu

## Loss
loss(x, y) = logitcrossentropy(model(x), y)
accuracy(x, y) = mean(onecold(cpu(model(x))) .== onecold(cpu(y)))


## Training
ps = Flux.params(model)
train_data = [(train_X, train_y)]
opt = ADAM(0.01)
evalcb() = @show(accuracy(train_X, train_y))

@epochs epochs Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
