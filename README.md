# GeometricFlux.jl

<p align="center">
<img width="400px" src="https://github.com/yuehhua/GeometricFlux.jl/raw/master/logos/logo.png"/>
</p>

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://yuehhua.github.io/GeometricFlux.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://yuehhua.github.io/GeometricFlux.jl/dev)
[![Build Status](https://travis-ci.org/yuehhua/GeometricFlux.jl.svg?branch=master)](https://travis-ci.org/yuehhua/GeometricFlux.jl)
[![codecov](https://codecov.io/gh/yuehhua/GeometricFlux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/yuehhua/GeometricFlux.jl)
[![pipeline status](https://gitlab.com/JuliaGPU/GeometricFlux-jl/badges/master/pipeline.svg)](https://gitlab.com/JuliaGPU/GeometricFlux-jl/commits/master)
[![coverage report](https://gitlab.com/JuliaGPU/GeometricFlux-jl/badges/master/coverage.svg)](https://gitlab.com/JuliaGPU/GeometricFlux-jl/commits/master)

GeometricFlux is a geometric deep learning library for [Flux](https://github.com/FluxML/Flux.jl). This library aims to be compatible with packages from [JuliaGraphs](https://github.com/JuliaGraphs) ecosystem and have support of CUDA GPU acceleration with [CUDA](https://github.com/JuliaGPU/CUDA.jl). Message passing scheme is implemented as a flexbile framework and fused with Graph Network block scheme. GeometricFlux is compatible with other packages that are composable with Flux.

Suggestions, issues and pull requsts are welcome.

## Installation

```
]add GeometricFlux
]add GraphSignals@0.1.1
```

## Features

* Extend Flux deep learning framework in Julia and compatible with Flux layers.
* Support of CUDA GPU with CUDA.jl
* Integrate with existing JuliaGraphs ecosystem
* Support generic graph neural network architectures
* Variable graph inputs are supported. You use it when diverse graph structures are prepared as inputs to the same model.
* Integrate GNN benchmark datasets (WIP)

## Graph convolutional layers

Construct GCN layer:

```
graph = # can be adj_mat, adj_list, simple_graphs...
GCNConv([graph, ]input_dim=>output_dim, relu)
```

## Use it as you use Flux

```
model = Chain(GCNConv(g, 1024=>512, relu),
              Dropout(0.5),
              GCNConv(g, 512=>128),
              Dense(128, 10),
              softmax)
## Loss
loss(x, y) = logitcrossentropy(model(x), y)
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

## Training
ps = Flux.params(model)
train_data = [(train_X, train_y)]
opt = ADAM(0.01)
evalcb() = @show(accuracy(train_X, train_y))

Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
```
