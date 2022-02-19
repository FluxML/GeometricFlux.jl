# Semi-supervised Learning with Graph Convolution Networks (GCN)

Graph convolution networks (GCN) have been considered as the first step to graph neural networks (GNN). This example will go through how to train a vanilla GCN.

## Semi-supervised Learning in Graph Neural Networks

The semi-supervised learning task defines a learning by given features and labels for only partial nodes in a graph. We train features and labels for partial nodes, and test the model for another partial nodes in graph.

## Node Classification task

In this task, we learn a node classification task which learns a model to predict labels for each node in a graph. In GCN network, node features are given and the model outputs node labels.

## Step 1: Load Dataset

GeometricFlux provides planetoid dataset in `GeometricFlux.Datasets`, which is provided by GraphMLDatasets. Planetoid dataset has three sub-datasets: Cora, Citeseer, PubMed. We demonstrate Cora dataset in this example. `traindata` provides the functionality for loading training data from various kinds of datasets. Dataset can be specified by the first argument, and the second for sub-datasets.

```julia
using GeometricFlux.Datasets

train_X, train_y = traindata(Planetoid(), :cora)
```

`traindata` returns a pre-defined training features and labels. These features are node features.

```julia
train_X, train_y = map(x->Matrix(x), traindata(Planetoid(), :cora))
```

We can load graph from `graphdata`, and the graph is preprocessed into `SimpleGraph` type, which is provided by Graphs.

```julia
g = graphdata(Planetoid(), :cora)
train_idx = train_indices(Planetoid(), :cora)
```

We need node indices to index a subgraph from original graph. `train_indices` gives node indices for training.

## Step 2: Wrapping Graph and Features into `FeaturedGraph`

`FeaturedGraph` is a container for holding a graph, node features, edge features and global features. It is provided by GraphSignals. To wrap graph and node features into `FeaturedGraph`, graph `g` should be placed as the first argument and `nf` is to specify node features.

```julia
using GraphSignals

FeaturedGraph(g, nf=train_X)
```

If we want to get a subgraph from a `FeaturedGraph` object, we call `subgraph` and provide node indices `train_idx` as second argument.

```julia
subgraph(FeaturedGraph(g, nf=train_X), train_idx)
```

## Step 3: Build a GCN model

A GCn model is composed of two layers of `GCNConv` and the activation function for first layer is `relu`. In the middle, a `Dropout` layer is placed. We need a `GraphParallel` to integrate with regular Flux layer, and it specifies node features go to `node_layer=Dropout(0.5)`.

```julia
model = Chain(
    GCNConv(input_dim=>hidden_dim, relu),
    GraphParallel(node_layer=Dropout(0.5)),
    GCNConv(hidden_dim=>target_dim),
    node_feature,
)
```

Since the model input is a `FeaturedGraph` object, the model output a `FeaturedGraph` object as well. In the end of model, we get node features out from a `FeaturedGraph` object using `node_feature`.

## Step 4: Loss Functions and Accuracy

Then, since it is a node classification task, we define the model loss by `logitcrossentropy`, and a L2 regularization is used. In the vanilla GCN, only first layer is applied to L2 regularization and can be adjusted by hyperparameter `λ`.

```julia
l2norm(x) = sum(abs2, x)

function model_loss(model, λ, batch)
    loss = 0.f0
    for (x, y) in batch
        loss += logitcrossentropy(model(x), y)
        loss += λ*sum(l2norm, Flux.params(model[1]))
    end
    return loss
end
```

Accuracy for a batch and for data loader are provided.

```julia
function accuracy(model, batch::AbstractVector)
    return mean(mean(onecold(softmax(cpu(model(x)))) .== onecold(cpu(y))) for (x, y) in batch)
end

accuracy(model, loader::DataLoader, device) = mean(accuracy(model, batch |> device) for batch in loader)
```

## Step 5: Training GCN Model

We train the model with the same process as training a Flux model.

```julia
train_loader, test_loader = load_data(:cora, args.batch_size)

# optimizer
opt = ADAM(args.η)
    
# parameters
ps = Flux.params(model)

# training
train_steps = 0
@info "Start Training, total $(args.epochs) epochs"
for epoch = 1:args.epochs
    @info "Epoch $(epoch)"

    for batch in train_loader
        grad = gradient(() -> model_loss(model, args.λ, batch |> device), ps)
        Flux.Optimise.update!(opt, ps, grad)
        train_steps += 1
    end
end
```

So far, we complete a basic tutorial for training a GCN model!

For the complete example, please check the script `examples/semisupervised_gcn.jl`.

## Acceleration by Pre-computing Normalized Adjacency Matrix

The training process can be slow in this example. Since we place the graph and features together in `FeaturedGraph` object, `GCNConv` will need to compute a normalized adjacency matrix in the training process. This behavior will lead to long training time. We can accelerate training process by pre-compute normalized adjacency matrix for all `FeaturedGraph` objects. To do so, we can call the following function and it will compute normalized adjacency matrix for `fg` before training. This will reduce the training time.

```julia
GraphSignals.normalized_adjacency_matrix!(fg)
```

Since the normalized adjacency matrix is used in `GCNConv`, we could pre-compute normalized adjacency matrix for it. If a layer doesn't require a normalized adjacency matrix, this step will lead to error.
