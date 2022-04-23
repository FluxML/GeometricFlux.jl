# Batch Learning

## Batch Learning for Variable Graph Strategy

Batch learning for variable graph strategy can be prepared as follows:

```julia
train_data = [(FeaturedGraph(g, nf=train_X), train_y) for _ in 1:N]
train_batch = Flux.batch(train_data)
```

It batches up [`FeaturedGraph`](@ref) objects into specified mini-batch. A batch is passed to a GNN model and trained/inferred one by one. It is hard for [`FeaturedGraph`](@ref) objects to train or infer in real batch for GPU.

## Batch Learning for Static Graph Strategy

A efficient batch learning should use static graph strategy. Batch learning for static graph strategy can be prepared as follows:

```julia
train_data = (repeat(train_X, outer=(1,1,N)), repeat(train_y, outer=(1,1,N)))
train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
```

An efficient batch learning should feed array to a GNN model. In the example, the mini-batch dimension is the third dimension for `train_X` array. The `train_X` array is split by `DataLoader` into mini-batches and feed a mini-batch to GNN model at a time. This strategy leverages the advantage of GPU training by accelerating training GNN model in a real batch learning.
