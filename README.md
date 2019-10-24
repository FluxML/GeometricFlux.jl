# GeometricFlux.jl

<p align="center">
<img width="400px" src="https://github.com/yuehhua/GeometricFlux.jl/raw/master/logos/logo.png"/>
</p>

[![Build Status](https://travis-ci.org/yuehhua/GeometricFlux.jl.svg?branch=master)](https://travis-ci.org/yuehhua/GeometricFlux.jl)
[![codecov](https://codecov.io/gh/yuehhua/GeometricFlux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/yuehhua/GeometricFlux.jl)

GeometricFlux is a geometric deep learning library for [Flux](https://github.com/FluxML/Flux.jl). This library aims to be compatible with packages from [JuliaGraphs](https://github.com/JuliaGraphs) ecosystem and have support of CUDA GPU acceleration with [CuArrays](https://github.com/JuliaGPU/CuArrays.jl). Message passing scheme is implemented as a flexbile framework and fused with Graph Network block scheme. GeometricFlux is compatible with other packages that are composable with Flux.

Suggestions, issues and pull requsts are welcome.

Note: Flux, Zygote, ZygoteRules, IRTools, CuArrays should use master branch.

## Installation

```
]add GeometricFlux
```

## Features

Construct layers from adjacency matrix or graph (maybe extend to other structures).
Input features (including vertex, edge or graph features) of neural network may not need a structure or type.
Labels or features for output of classification or regression are part of training data, they may not need a specific structure or type, too.

* Integration of JuliaGraphs
    * [x] Construct layer from SimpleGraph
    * [x] Construct layer from SimpleWeightedGraph
    * [x] Construct layer from Matrix
    * [ ] Support vertex/edge/graph features from MetaGraphs
* Layers
    * Convolution layers
        * [x] MessagePassing
        * [x] GCNConv
        * [x] GraphConv
        * [x] ChebConv
        * [x] GatedGraphConv
        * [x] GATConv
        * [x] EdgeConv
        * [x] Meta
    * Pooling layers
        * [x] GlobalPool
        * [ ] TopKPool
        * [ ] LocalPool
        * [x] sum/sub/prod/div/max/min/mean pool
    * Embedding layers
        * [x] InnerProductDecoder
* Models
    * [ ] VGAE
    * [x] GAE
* Internals
    * [x] use Zygote
    * [x] compatible with layers in Flux
    * [x] multi-threading scatter (i.e. add/sub/prod/div/max/min/mean)
* Datasets
* Storage
    * [ ] Benchmark JLD2, BSON
