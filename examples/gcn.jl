using GeometricFlux
using Flux
using Flux: onehotbatch, onecold, crossentropy, throttle
using JLD2
using StatsBase
using SparseArrays
using LightGraphs.SimpleGraphs

@load "data/cora_features.jld2" features
@load "data/cora_labels.jld2" labels
@load "data/cora_graph.jld2" g

num_nodes = 2708
num_features = 1433
target_catg = 7

## Preprocessing data
train_X = features  # dim: num_features * num_nodes
train_y = labels  # dim: target_catg * num_nodes

## Model
model = Chain(GCNConv(g, num_features=>1000, relu),
              GCNConv(g, 1000=>500, relu),
              GCNConv(g, 500=>7),
              softmax)
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

Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
