using GeometricFlux
using Flux
using Flux: onehotbatch, onecold, logitcrossentropy, throttle
using Flux: @epochs
using JLD2  # use v0.1.2
using Statistics: mean
using SparseArrays
using LightGraphs.SimpleGraphs
using LightGraphs: adjacency_matrix
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
train_X = features |> gpu  # dim: num_features * num_nodes
train_y = labels |> gpu  # dim: target_catg * num_nodes

## Model
model = Chain(GATConv(g, num_features=>hidden, heads=heads),
              Dropout(0.6),
              GATConv(g, hidden=>target_catg, heads=heads),
              softmax) |> gpu
# test model
# model(train_X)

## Loss
loss(x, y) = logitcrossentropy(model(x), y)

## Training
ps = Flux.params(model)
train_data = [(train_X, train_y)]
opt = ADAM(0.01)
evalcb() = @show(loss(train_X, train_y))

@epochs epochs Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
