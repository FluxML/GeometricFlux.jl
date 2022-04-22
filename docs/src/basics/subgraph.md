# Subgraph

## Subgraph of `FeaturedGraph`

A [`FeaturedGraph`](@ref) object can derive a subgraph from a selected subset of the vertices of the graph.

```julia
train_idx = train_indices(Planetoid(), :cora)
fg = FeaturedGraph(g)
fsg = subgraph(fg, train_idx)
```

A `FeaturedSubgraph` object is returned from [`subgraph`](@ref) by selected vertices `train_idx`.
