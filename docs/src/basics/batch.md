# Batch Learning

## Mini-batch Learning for [`FeaturedGraph`](@ref)

```@raw html
<figure>
    <img src="../assets/FeaturedGraph-support-DataLoader.svg" width="50%" alt="FeaturedGraph-support-DataLoader.svg" /><br>
    <figcaption><em>FeaturedGraph supports DataLoader.</em></figcaption>
</figure>
```

Batch learning for [`FeaturedGraph`](@ref) can be prepared as follows:

```julia
train_data = (FeaturedGraph(g, nf=train_X), train_y)
train_batch = DataLoader(train_data, batchsize=batch_size, shuffle=true)
```

[`FeaturedGraph`](@ref) now supports `DataLoader` and one can specify mini-batch to it.
A mini-batch is passed to a GNN model and trained/inferred in one [`FeaturedGraph`](@ref).

## Mini-batch Learning for array

```@raw html
<figure>
    <img src="../assets/cuda-minibatch.svg" width="50%" alt="cuda-minibatch.svg" /><br>
    <figcaption><em>Mini-batch learning on CUDA.</em></figcaption>
</figure>
```

Mini-batch learning for array can be prepared as follows:

```julia
train_loader = DataLoader((train_X, train_y), batchsize=batch_size, shuffle=true)
```

An array could be fed to a GNN model. In the example, the mini-batch dimension is the last dimension for `train_X` array. The `train_X` array is split by `DataLoader` into mini-batches and feed a mini-batch to GNN model at a time. This strategy leverages the advantage of GPU training by accelerating training GNN model in a real batch learning.
