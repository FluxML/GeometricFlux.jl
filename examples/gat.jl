using GeometricFlux
using Flux
using Flux: onehotbatch, onecold, logitcrossentropy, throttle
using Flux: @epochs
using JLD2
using Statistics: mean
using SparseArrays
using LinearAlgebra
using Graphs.SimpleGraphs
using Graphs: adjacency_matrix
using CUDA

@load "data/cora_features.jld2" features
@load "data/cora_labels.jld2" labels
@load "data/cora_graph.jld2" g

num_nodes = 2708
num_features = 1433
heads  = 8
hidden = 8
target_catg = 7
epochs = 10

## Preprocessing data
train_X = Matrix{Float32}(features) |> gpu  # dim: num_features * num_nodes
train_y = Matrix{Float32}(labels) |> gpu  # dim: target_catg * num_nodes
A = Matrix{Int}((adjacency_matrix(g) + I) .≥ 1)
fg = FeaturedGraph(A, :adjm)

## Model
model = Chain(GATConv(fg, num_features=>hidden, heads=heads),
              Dropout(0.6),
              GATConv(fg, hidden*heads=>target_catg, heads=heads, concat=false)
              ) |> gpu
# test model
@show model(train_X)

## Loss
loss(x, y) = logitcrossentropy(model(x), y)
accuracy(x, y) = mean(onecold(cpu(model(x))) .== onecold(cpu(y)))

# test loss
@show loss(train_X, train_y)

# test gradient
@show gradient(()->loss(train_X, train_y), Flux.params(model))

## Training
ps = Flux.params(model)
train_data = Flux.Data.DataLoader((train_X, train_y), batchsize=num_nodes)
opt = ADAM(0.01)
evalcb() = @show(accuracy(train_X, train_y))

@epochs epochs Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
