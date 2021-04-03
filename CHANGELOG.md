# Changelog

All notable changes to this project will be documented in this file.

## [0.7.6]

- Add dimensional check for each layer
- Support Flux up to v0.12
- Support CUDA up to v2.6
- Support Zygote up to v0.6

## [0.7.5]

- FeaturedGraph API change
- Refactor graph net and message passing framework
- Improve differentiability test
- Refactor GCNConv and ChebConv operator
- Fix bug in GATConv layer
- Update GAT example
- Cast testing data to Float32
- Support CUDA up to v2.2
- Support transpose input of a layer
- Replace Travis CI by Github Action CI

## [0.7.4]

- Adjust edge_index_table API for directed
- apply_batch_message as API
- Support CUDA v2.1
- Refactor
- Fix bug

## [0.7.3]

- Add bypass_graph
- Support FeaturedGraph as input graph for GCNConv
- Add node index for message/update function
- Add activation function for GraphConv
- Reexport GraphSignals
- Support FillArrays v0.10
- Bug fix

## [0.7.2]

- Differentiability test
- Refactor GN for differentiability
- Remove cache argument from layer
- Add docs
- Bump CUDA to v2.0
- Add paper

## [0.7.1]

- Add GraphMLDatasets as dependency to provide datasets

## [0.7.0]

- VGAE example available
- Add Planetoid and Cora dataset

## [0.6.3]

- GDE, GAE VGAE examples available
- Correct GCNConv show

## [0.6.2]

- Add FeatureSelector
- Correct ChebConv computation
- Make scaled_laplacian differentiable
- Add ScatterNNlib and GraphSignals as deps
- Improve GAT example
- Upgrade to CUDA
- Maintain Travis CI

## [0.6.1]

- Update to CUDA 1.2 and Flux 0.11
- Refactor graph-related API
- Improve learning rate in example

## [0.6.0]

- Rewrite graph network `GraphNet` and message passing `MessagePassing` framework
- Expand functionality of FeaturedGraph to support `node_feature`, `edge_feature` and `global_feature`
- Speed up ChebConv layer
- Speed up scatter functions
- Add graph index-related functions
- GCN example works and increase training stablility
- Fix show GCNConv
- Add more test for linear algebra
- Update cpu scatter benchmark plot and scripts

## [0.5.2]

- Add scaled Laplacian
- Support CuArrays v2.0 and Flux v0.10.4
- ChebConv, GraphConv, GATConv, GatedGraphConv and EdgeConv support FeaturedGraph
- Add SimpleWeightedGraphs and MetaGraphs as deps
- Fix broadcastly casting error

## [0.5.1]

- GCNConv layer supports FeaturedGraph (#34)
- Support linear algebra for FeaturedGraph
- Add `nv` API for FeaturedGraph
- Add LightGraphs as dependency
- Correct normalized laplacian type
- Fix bug in normalized_laplacian
- Fix Base.show on GCNConv
- Add docs (#35)

## [0.5.0]

- Support scatter operations for MArray (#32)
- Support GCNConv layer accepting graph input (#31)

## [0.4.0]

- Compatible with Julia v1.4 while not support before v1.3
- Not support old version CuArrays, CUDAnative and CUDAapi
- Improve performance of scatter operations for CPU and new benchmark (#29)
- Scatters support almost all Real numbers except Bool on CPU
- Add benchmark for scatter operations
- Implement TopKPool layer (#22)

## [0.3.0]

- Improve performance of scatter operations in both CPU/CUDA version
- Add benchmark result
- Add multihead GAT on graph support
- Move `pool_dim_check` to `Dims` constructor

## [0.2.0]

 - Available on Julia v1.2 and v1.3
 - Convolution layers works with CUDA
 - Provide scatter add, sub, mul, div, max, min, mean for CPU and CUDA
 - Provide pool add, sub, mul, div, max, min, mean for CPU and CUDA
 - Provide gradient of scatter add, sub, mul, div, max, min, mean for CPU and CUDA
 - Provide gradient of pool add, sub, mul, div, max, min, mean for CPU and CUDA
 - Provide gather
 - Provide good abstract for graph network block
 - Integrate message passing scheme and graph network block
 - Add logo
 - Add docs
 - Add layer docs and Base.show
 - Provide dynamically change graph in runtime
 - Provide GlobalPool layer
