using GeometricFlux
using Flux
using Flux: onehotbatch, onecold, logitcrossentropy, throttle
using Flux: @epochs
using JLD2  # use v0.1.2
using Statistics: mean
using SparseArrays
using LightGraphs.SimpleGraphs
using LightGraphs: adjacency_matrix
using CuArrays

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
train_X = Float32.(features) |> gpu  # dim: num_features * num_nodes
train_y = Float32.(labels) |> gpu  # dim: target_catg * num_nodes
adj_mat = Matrix{Float32}(adjacency_matrix(g)) |> gpu

## Model
model = Chain(GATConv(g, num_features=>hidden, heads=heads),
              Dropout(0.6),
              GATConv(g, hidden*heads=>target_catg, heads=heads, concat=false),
              softmax) |> gpu
# test model
# model(train_X)

## Loss
loss(x, y) = logitcrossentropy(model(x), y)
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

# test gradient
# gradient(X -> loss(X, train_y), train_X)

## Training
ps = Flux.params(model)
train_data = [(train_X, train_y)]
opt = ADAM(0.05)
evalcb() = @show(accuracy(train_X, train_y))

@epochs epochs Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
