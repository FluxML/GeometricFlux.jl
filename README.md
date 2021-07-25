# GeometricFlux.jl

<p align="center">
<img width="400px" src="https://github.com/FluxML/GeometricFlux.jl/raw/master/logos/logo.png"/>
</p>

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://fluxml.ai/GeometricFlux.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://fluxml.ai/GeometricFlux.jl/dev)
![](https://github.com/FluxML/GeometricFlux.jl/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/FluxML/GeometricFlux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/FluxML/GeometricFlux.jl)

GeometricFlux is a geometric deep learning library for [Flux](https://github.com/FluxML/Flux.jl). This library aims to be compatible with packages from [JuliaGraphs](https://github.com/JuliaGraphs) ecosystem and have support of CUDA GPU acceleration with [CUDA](https://github.com/JuliaGPU/CUDA.jl). Message passing scheme is implemented as a flexbile framework and fused with Graph Network block scheme. GeometricFlux is compatible with other packages that are composable with Flux.

Suggestions, issues and pull requsts are welcome.

## Installation

```julia
]add GeometricFlux
```

## Features

* Extend Flux deep learning framework in Julia and compatible with Flux layers.
* Support of CUDA GPU with CUDA.jl
* Integrate with existing JuliaGraphs ecosystem
* Support generic graph neural network architectures
* Variable graph inputs are supported. You use it when diverse graph structures are prepared as inputs to the same model.
* Integrate GNN benchmark datasets (WIP)

## Featured Graphs

GeometricFlux handles graph data (the topology plus node/vertex/graph features)
thanks to the type `FeaturedGraph`.

A `FeaturedGraph` can be constructed out of 
adjacency matrices, adjacency lists, LightGraphs' types...

```julia
fg = FeaturedGraph(adj_list)   
```
## Graph convolutional layers

Construct a GCN layer:

```julia
GCNConv([fg,] input_dim => output_dim, relu)
```

## Use it as you use Flux

```julia
model = Chain(GCNConv(fg, 1024 => 512, relu),
              Dropout(0.5),
              GCNConv(fg, 512 => 128),
              Dense(128, 10))
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
