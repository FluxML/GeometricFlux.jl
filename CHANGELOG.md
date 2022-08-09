# Changelog

All notable changes to this project will be documented in this file.

## [0.13.4]

- support GraphSignals to 0.7

## [0.13.3]

- update doc for `FeaturedGraph`

## [0.13.2]

- fix doc

## [0.13.1]

- `GraphParallel` support `positional_layer`

## [0.13.0]

- implement `EEquivGraphConv` layer with nested design
- support positional encoding from GraphSignals
- add `LSPE`

## [0.12.4]

- replace `ADAM` as `Adam`

## [0.12.3]

- update doc for graph network

## [0.12.2]

- replace `Zygote.ignore` as `ChainRulesCore.ignore_derivatives`

## [0.12.1]

- remove @deprecate

## [0.12.0]

- add roadmap
- add `SAGEConv` layer
- support dynamic graph update

## [0.11.1]

- fix link

## [0.11.0]

- Adds GATv2 layer
- add DeepSet model and digit sum example
- fix GAT example
- add tutorials
- replace `Flux.destructure` by `Optimisers.destructure`

## [0.10.1]

- fix VGAE example and correct precision
- implement new message-passing scheme, including `GraphConv`, `GATConv`, `GatedGraphConv`, `EdgeConv`, `GINConv` and `CGConv` layers
- fix tests for `GraphNet`
- add `WithGraph` for `Chain`

## [0.10.0]

- update docs and add defining GNN layer to doc
- update GAE example
- fix neural GDE example

## [0.9.0]

- add semisupervised gcn and gcn with fixed graph example
- implement new GCNConv
- add node2vec
- bug fix

## [0.8.0]

- correct GCNConv with normalized_adjacency_matrix
- add L2 regularization to gcn example
- migrate Graphs, GraphSignals, GraphLaplacians and examples
- resolve gradient bug for GatedGraphConv

## [0.7.7]

- drop support of julia v1.4 and v1.5
- support CUDA v3.3
- support Flux v0.12
- fix stable doc
- add benchmark script
- migrate scatter to NNlib
- make gradient of GatedGraphConv available
- Implement GINConv layer. (#186)
- check consistency for vertex or edge number between graph and features
- add manual for pooling layers and bypass_graph
- deprecate FeatureSelector
- not export GraphNetwork and MessagePassing APIs
- new implementation for message-passing scheme

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
