# Subgraph

```julia
train_idx = train_indices(Planetoid(), :cora)
fg = FeaturedGraph(g)
fsg = subgraph(fg, train_idx)
layer = WithGraph(fsg, GCNConv(in_channel=>out_channel), ) |> gpu
train_X = train_X |> Matrix |> gpu
H = layer(train_X)
```
