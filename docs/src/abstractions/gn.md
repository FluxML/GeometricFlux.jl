# Graph network block

Graph network (GN) is a more generic model for graph neural network. It describes an update order: edge, node and then global. There are three corresponding update functions for edge, node and then global, respectively. Three update functions return their default values as follow:

```
update_edge(gn, e, vi, vj, u) = e
update_vertex(gn, ē, vi, u) = vi
update_global(gn, ē, v̄, u) = u
```

Information propagation between different levels are achieved by aggregate functions. Three aggregate functions `aggregate_neighbors`, `aggregate_edges` and `aggregate_vertices` are defined to aggregate states.

GN block is realized into a abstract type `GraphNet`. User can make a subtype of `GraphNet` to customize GN block. Thus, a GN block is defined as a layer in GNN. `MessagePassing` is a subtype of `GraphNet`.

## Update functions

`update_edge` acts as the first update function to apply to edge states. It takes edge state `e`, node `i` state `vi`, node `j` state `vj` and global state `u`. It is expected to return a feature vector for new edge state. `update_vertex` updates nodes state by taking aggregated edge state `ē`, node `i` state `vi` and global state `u`. It is expected to return a feature vector for new node state. `update_global` updates global state with aggregated information from edge and node. It takes aggregated edge state `ē`, aggregated node state `v̄` and global state `u`. It is expected to return a feature vector for new global state. User can define their own behavior by overriding update functions.

## Aggregate functions

An aggregate function `aggregate_neighbors` aggregates edge states for edges incident to some node `i` into node-level information. Aggregate function `aggregate_edges` aggregates all edge states into global-level information. The last aggregate function `aggregate_vertices` aggregates all vertex states into global-level information. It is available for assigning aggregate function by assigning aggregate operations to `propagate` function.

```
propagate(gn, fg::FeaturedGraph, naggr=nothing, eaggr=nothing, vaggr=nothing)
```

`naggr`, `eaggr` and `vaggr` are arguments for `aggregate_neighbors`, `aggregate_edges` and `aggregate_vertices`, respectively. Available aggregate functions are assigned by following symbols to them: `:add`, `:sub`, `:mul`, `:div`, `:max`, `:min` and `:mean`.
