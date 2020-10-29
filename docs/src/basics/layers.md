# Building graph neural networks

Building GNN is as simple as building neural network in Flux. The syntax here is the same as Flux. `Chain` is used to stack layers into a GNN. A simple example is shown here:

```
model = Chain(GCNConv(adj_mat, feat=>h1),
              GCNConv(adj_mat, h1=>h2, relu))
```

`GCNConv` is used for layer construction for neural network. The first argument `adj_mat` is the representation of a graph in form of adjacency matrix. The feature dimension in first layer is mapped from `feat` to `h1`. In second layer, `h1` is then mapped to `h2`. Default activation function is given as identity if it is not specified by users.

## Customize layers

Customizing your own GNN layers are the same as customizing layers in Flux. You may want to reference [Flux documentation](https://fluxml.ai/Flux.jl/stable/models/basics/#Building-Layers-1).
