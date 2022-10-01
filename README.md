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

* Extending Flux deep learning framework in Julia and seamlessly integration with regular Flux layers.
* Support of CUDA GPU with CUDA.jl and mini-batched training leveraging advantages of GPU
* Integration with existing JuliaGraphs ecosystem
* Support Message-passing and graph network architectures
* Support of static graph and variable graph strategy. Variable graph strategy is useful when training the model over diverse graph structures.
* Integration of GNN benchmark datasets
* Support dynamic graph update towards manifold learning 2.0

### Featured Graphs

GeometricFlux handles graph data (the topology plus node/vertex/graph features)
thanks to `FeaturedGraph` type.

A `FeaturedGraph` can be constructed from various graph structures, including
adjacency matrices, adjacency lists, Graphs' types...

```julia
fg = FeaturedGraph(adj_list)
```

### Graph convolutional layers

Construct a GCN layer:

```julia
GCNConv(input_dim => output_dim, relu)
```

## Use it as you use Flux

```julia
model = Chain(
    WithGraph(fg, GCNConv(fg, 1024 => 512, relu)),
    Dropout(0.5),
    WithGraph(fg, GCNConv(fg, 512 => 128)),
    Dense(128, 10)
)
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

## Roadmap

To achieve geometric deep learning raised by Bronstein *et al*, 5G fields of deep learning models will be supported in GeometricFlux.jl. For details, you could check the [geometric deep learning official website](https://geometricdeeplearning.com/).

5(+1)G including the following fields:

* **Graphs** and Sets
  * including classical GNN models and networks over sets.
  * Transformer models are regard as a kind of GNN with complete graph, and you can check [chengchingwen/Transformers.jl](https://github.com/chengchingwen/Transformers.jl) for more details.
* **Grids** and Euclidean spaces
  * including classical convolutional neural networks, multi-layer perceptrons etc.
  * for operators over functional spaces of regular grid, you can check [SciML/NeuralOperators.jl](https://github.com/SciML/NeuralOperators.jl) for more details.
* **Groups** and Homogeneous spaces
  * including a series of equivariant/invariant models.
* **Geodesics** and Manifolds
* **Gauges** and Bundles
* **Geometric algebra**

## Discussions

It's welcome to have direct discussions in #graphnet channel or in #flux-bridged channel on slack. For usage issues, it's welcome to post your minimal working examples (MWE) on [Julia discourse](https://discourse.julialang.org/) and then tag maintainer `@yuehhua`.
