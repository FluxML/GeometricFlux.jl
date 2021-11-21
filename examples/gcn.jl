using GeometricFlux
using GraphSignals
using Flux
using Flux: onehotbatch, onecold, logitcrossentropy, throttle
using Flux: @epochs
using JLD2
using Statistics
using SparseArrays
using Graphs.SimpleGraphs
using CUDA

@load "data/cora_features.jld2" features
@load "data/cora_labels.jld2" labels
@load "data/cora_graph.jld2" g

num_nodes = 2708
num_features = 1433
hidden = 16
target_catg = 7
epochs = 200
λ = 5e-4

## Preprocessing data
train_X = Matrix{Float32}(features) |> gpu  # dim: num_features * num_nodes
train_y = Matrix{Float32}(labels) |> gpu  # dim: target_catg * num_nodes
fg = FeaturedGraph(g)  # pass to gpu together in model layers

## Model
model = Chain(GCNConv(fg, num_features=>hidden, relu),
              Dropout(0.5),
              GCNConv(fg, hidden=>target_catg),
              ) |> gpu;
# do not show model architecture, showing CuSparseMatrix will trigger errors

## Loss
l2norm(x) = sum(abs2, x)
# cross entropy with first layer L2 regularization 
loss(x, y) = logitcrossentropy(model(x), y) + λ*sum(l2norm, Flux.params(model[1]))
accuracy(x, y) = mean(onecold(softmax(cpu(model(x)))) .== onecold(cpu(y)))


## Training
ps = Flux.params(model)
train_data = [(train_X, train_y)]
opt = ADAM(0.01)
evalcb() = @show(accuracy(train_X, train_y))

@epochs epochs Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
