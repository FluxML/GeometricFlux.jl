# GeometricFlux: The Geometric Deep Learning Library in Julia

GeometricFlux is a framework for geometric deep learning/machine learning.

* It extends Flux machine learning library for geometric deep learning.
* It supports of CUDA GPU with CUDA.jl
* It integrates with JuliaGraphs ecosystems.
* It supports generic graph neural network architectures (i.g. message passing scheme and graph network block)
* It contains built-in GNN benchmark datasets (provided by GraphMLDatasets)

## Installation

```
] add GeometricFlux
```

## Quick start

The basic graph convolutional network (GCN) is constructed as follow.

```
model = Chain(GCNConv(adj_mat, num_features=>hidden, relu),
              GCNConv(adj_mat, hidden=>target_catg),
              softmax)
```

### Load dataset

Load cora dataset from GeometricFlux.

```
using JLD2
using SparseArrays

@load joinpath(pkgdir(GeometricFlux), "data/cora_features.jld2") features
@load joinpath(pkgdir(GeometricFlux), "data/cora_labels.jld2") labels
@load joinpath(pkgdir(GeometricFlux), "data/cora_graph.jld2") g
```

### Training/testing data

Data is stored in sparse array, thus, we have to convert it into normal array.

```
train_X = Matrix{Float32}(features)
train_y = Matrix{Float32}(labels)
adj_mat = Matrix{Float32}(adjacency_matrix(g))
```

### Loss function

```
loss(x, y) = logitcrossentropy(model(x), y)
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
```

### Training

```
ps = Flux.params(model)
train_data = [(train_X, train_y)]
opt = ADAM()
evalcb() = @show(accuracy(train_X, train_y))

@epochs epochs Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
```

### Logs

```
[ Info: Epoch 1
accuracy(train_X, train_y) = 0.11669128508124077
[ Info: Epoch 2
accuracy(train_X, train_y) = 0.19608567208271788
[ Info: Epoch 3
accuracy(train_X, train_y) = 0.3098227474150665
[ Info: Epoch 4
accuracy(train_X, train_y) = 0.387370753323486
[ Info: Epoch 5
accuracy(train_X, train_y) = 0.44645494830132937
[ Info: Epoch 6
accuracy(train_X, train_y) = 0.46824224519940916
[ Info: Epoch 7
accuracy(train_X, train_y) = 0.48892171344165436
[ Info: Epoch 8
accuracy(train_X, train_y) = 0.5025849335302807
[ Info: Epoch 9
accuracy(train_X, train_y) = 0.5151403249630724
[ Info: Epoch 10
accuracy(train_X, train_y) = 0.5291728212703102
[ Info: Epoch 11
accuracy(train_X, train_y) = 0.543205317577548
[ Info: Epoch 12
accuracy(train_X, train_y) = 0.5550221565731167
[ Info: Epoch 13
accuracy(train_X, train_y) = 0.5638847858197932
[ Info: Epoch 14
accuracy(train_X, train_y) = 0.5657311669128509
[ Info: Epoch 15
accuracy(train_X, train_y) = 0.5749630723781388
[ Info: Epoch 16
accuracy(train_X, train_y) = 0.5834564254062038
[ Info: Epoch 17
accuracy(train_X, train_y) = 0.5919497784342689
[ Info: Epoch 18
accuracy(train_X, train_y) = 0.5978581979320532
[ Info: Epoch 19
accuracy(train_X, train_y) = 0.6019202363367799
[ Info: Epoch 20
accuracy(train_X, train_y) = 0.6067208271787297
```

Check `examples/gcn.jl` for details.
