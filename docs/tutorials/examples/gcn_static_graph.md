---
title: GCN with Static Graph
cover: assets/logo.svg
id: gcn_static_graph
---

# GCN with Static Graph

In the tutorial for semi-supervised learning with GCN, variable graphs are provided to GNN from `FeaturedGraph`, which contains a graph and node features. Each `FeaturedGraph` object can contain different graph and different node features, and can be train on the same GNN model. However, variable graph doesn't have the proper form of graph structure with respect to GNN layers and this lead to inefficient training/inference process. Static graph strategy can be used to train a GNN model with the same graph structure in GeometricFlux.

## Static Graph

A static graph is given to a layer by `WithGraph` syntax. `WithGraph` wrap a `FeaturedGraph` object and a GNN layer as first and second arguments, respectively.

```julia
fg = FeaturedGraph(graph)
WithGraph(fg, GCNConv(1024=>256, relu))
```

This way, we can customize by binding different graph to certain layer and the layer will specialize graph to a required form. For example, a `GCNConv` layer requires graph in the form of normalized adjacency matrix. Once the graph is bound to a `GCNConv` layer, it transforms graph into normalized adjacency matrix and stores in `WithGraph` object. It accelerates training or inference by avoiding calculating transformations. The features in `FeaturedGraph` object in `WithGraph` are not used in any layer or model training or inference.

## Array in, Array out

With this approach, a GNN layer accepts features in array. It takes an array as input and outputs array. Thus, a GNN layer wrapped with `WithGraph` should accept a feature array, just like regular deep learning model.

## Batch Learning

Since features are in the form of array, they can be batched up for batched learning. We will demonstrate how to achieve these goals.

## Step 1: Load Dataset

Different from loading datasets in semi-supervised learning example, we use `alldata` for supervised learning here and `padding=true` is added in order to padding features from partial nodes to pseudo-full nodes. A padded features contains zeros in the nodes that are not supposed to be train on.

```julia
data = dataset[1].node_data
X, y = data.features, onehotbatch(data.targets, 1:7)
train_idx, test_idx = data.train_mask, data.val_mask
train_X, train_y = repeat(X, outer=(1,1,train_repeats)), repeat(y, outer=(1,1,train_repeats))
```

We need graph and node indices for training as well.

```julia
s, t = dataset[1].edge_index
g = Graphs.Graph(dataset[1].num_nodes)
for (i, j) in zip(s, t)
    Graphs.add_edge!(g, i, j)
end
fg = FeaturedGraph(g)
```

## Step 2: Batch up Features and Labels

In order to make batch learning available, we separate graph and node features. We don't subgraph here. Node features are batched up by repeating node features here for demonstration, since planetoid dataset doesn't have batched settings. Different repeat numbers can be specified by `train_repeats` and `train_repeats`.

```julia
train_loader = DataLoader((train_X, train_y), batchsize=batch_size, shuffle=true)
```

## Step 3: Build a GCN model

Here comes to building a GCN model. We build a model as building a regular Flux model but just wrap `GCNConv` layer with `WithGraph`.

```julia
model = Chain(
    WithGraph(fg, GCNConv(args.input_dim=>args.hidden_dim, relu)),
    Dropout(0.5),
    WithGraph(fg, GCNConv(args.hidden_dim=>args.target_dim)),
)
```

## Step 4: Loss Functions and Accuracy

Almost all codes are the same as in semi-supervised learning example, except that indices for subgraphing are needed to get partial features out for calculating loss.

```julia
l2norm(x) = sum(abs2, x)

function model_loss(model, λ, X, y, idx)
    loss = logitcrossentropy(model(X)[:,idx,:], y[:,idx,:])
    loss += λ*sum(l2norm, Flux.params(model[1]))
    return loss
end
```

And the accuracy measurement also needs indices.

```julia
function accuracy(model, X::AbstractArray, y::AbstractArray, idx)
    return mean(onecold(softmax(cpu(model(X))[:,idx,:])) .== onecold(cpu(y)[:,idx,:]))
end

accuracy(model, loader::DataLoader, device, idx) = mean(accuracy(model, X |> device, y |> device, idx) for (X, y) in loader)
```

## Step 5: Training GCN Model

```julia
train_loader, test_loader, fg, train_idx, test_idx = load_data(:cora, args.batch_size)

# optimizer
opt = ADAM(args.η)

# parameters
ps = Flux.params(model)

# training
train_steps = 0
@info "Start Training, total $(args.epochs) epochs"
for epoch = 1:args.epochs
    @info "Epoch $(epoch)"

    for (X, y) in train_loader
        X, y, device_idx = X |> device, y |> device, train_idx |> device
        grad = gradient(() -> model_loss(model, args.λ, X, y, device_idx), ps)
        Flux.Optimise.update!(opt, ps, grad)
        train_steps += 1
    end
end
```

Now we could just train the GCN model directly!
