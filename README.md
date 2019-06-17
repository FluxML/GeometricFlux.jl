# GeometricFlux.jl

## Roadmap

* Start
    * [ ] Establish a simple example of GNN
* Integration of JuliaGraphs
    * [ ] Construct layer from SimpleGraph
    * [ ] Construct layer from WeightedSimpleGraph
    * [ ] Construct layer from Matrix
    * Support vertex/edge/graph features from MetaGraphs
* Layers
    * Convolution layers
        * [ ] MessagePassing
        * [ ] GCNConv
        * [ ] GraphConv
        * [ ] ChebConv
        * [ ] GatedGraphConv
        * [ ] GATConv
        * [ ] EdgeConv
        * [ ] Meta
    * Pooling layers
        * [ ] GlobalPool
        * [ ] TopKPool
        * [ ] MaxPool
        * [ ] MeanPool
    * Embedding layers
        * [ ] InnerProductDecoder
* Networks
    * [ ] VGAE
* Datasets
    * Benchmark JLD2, BSON
