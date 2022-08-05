---
title: Graph Attention Network
cover: assets/logo.svg
id: gat
---

# Graph Attention Network

Graph attention network (GAT) belongs to the message-passing network family, and it queries node feature over its neighbor features and generates result as layer output.

## Step 1: Load Dataset

We load dataset from Planetoid dataset. Here cora dataset is used.

```julia
data = dataset[1].node_data
X, y = data.features, onehotbatch(data.targets, 1:7)
train_idx, test_idx = data.train_mask, data.val_mask
```

## Step 2: Batch up Features and Labels

Just batch up features as usual.

```julia
s, t = dataset[1].edge_index
g = Graphs.Graph(dataset[1].num_nodes)
for (i, j) in zip(s, t)
    Graphs.add_edge!(g, i, j)
end

add_all_self_loops!(g)
fg = FeaturedGraph(g)
train_X, train_y = repeat(X, outer=(1,1,train_repeats)), repeat(y, outer=(1,1,train_repeats))
train_loader = DataLoader((train_X, train_y), batchsize=batch_size, shuffle=true)
```

Notably, self loop for all nodes are needed for GAT model.

## Step 3: Build a GAT model

```julia
model = Chain(
    WithGraph(fg, GATConv(args.input_dim=>args.hidden_dim, heads=args.heads)),
    Dropout(0.6),
    WithGraph(fg, GATConv(args.hidden_dim*args.heads=>args.target_dim, heads=args.heads, concat=false)),
) |> device
```

To note that a `GATConv` with `concat=true` will accumulates `heads` onto feature dimension. Thus, in the next layer, we should use `args.hidden_dim*args.heads`. In the final layer of a network, a `GATConv` layer should be assigned with `concat=false` to average over each heads.


## Step 4: Loss Functions and Accuracy

Cross entropy loss is used as loss function and accuracy is used to evaluate the model.

```julia
model_loss(model, X, y, idx) =
    logitcrossentropy(model(X)[:,idx,:], y[:,idx,:])
```

```julia
accuracy(model, X::AbstractArray, y::AbstractArray, idx) =
    mean(onecold(softmax(cpu(model(X))[:,idx,:])) .== onecold(cpu(y)[:,idx,:])
```


## Step 5: Training GAT Model

```julia
# ADAM optimizer
opt = ADAM(args.η)

# parameters
ps = Flux.params(model)

# training
@info "Start Training, total $(args.epochs) epochs"
for epoch = 1:args.epochs
    @info "Epoch $(epoch)"

    for (X, y) in train_loader
        X, y, device_idx = X |> device, y |> device, train_idx |> device
        loss, back = Flux.pullback(() -> model_loss(model, X, y, device_idx), ps)
        train_acc = accuracy(model, train_loader, device, train_idx)
        test_acc = accuracy(model, test_loader, device, test_idx)
        grad = back(1f0)
        Flux.Optimise.update!(opt, ps, grad)
    end
end
```

For a complete example, please check [examples/gat.jl](https://github.com/FluxML/GeometricFlux.jl/blob/master/examples/gat.jl).
