using GeometricFlux
using Flux
using Flux: onehotbatch, onecold, crossentropy, throttle
using JLD2  # use v0.1.2
using StatsBase
using SparseArrays
using LightGraphs.SimpleGraphs
using CuArrays

@load "data/cora_features.jld2" features
@load "data/cora_labels.jld2" labels
@load "data/cora_graph.jld2" g

num_nodes = 2708
num_features = 1433
target_catg = 7
epochs = 10

## Preprocessing data
train_X = features |> gpu  # dim: num_features * num_nodes
train_y = labels |> gpu  # dim: target_catg * num_nodes

## Model
model = Chain(GCNConv(g, num_features=>1000, relu),
              GCNConv(g, 1000=>500, relu),
              Dense(500, 7),
              softmax) |> gpu
# test model
# model(train_X)

## Loss
loss(x, y) = crossentropy(model(x), y)
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

## Training
ps = Flux.params(model)
train_data = [(train_X, train_y)]
opt = ADAM(0.01)
evalcb() = @show(accuracy(train_X, train_y))

for i = 1:epochs
    Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
end
