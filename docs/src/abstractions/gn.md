# Graph network block

Graph network (GN) is a more generic model for graph neural network. For details, a graph network block is defined as follow:

```math
\begin{aligned}
    e_{ij}^{\prime} &= \phi^{e}(e_{ij}, v_i, v_j, u) \\
    v_{i}^{\prime} &= \phi^{v}(\bar{e}_{i}^{\prime}, v_i, u) \\
    u^{\prime} &= \phi^{u}(\bar{e}^{\prime}, \bar{v}^{\prime}, u)
\end{aligned}
\ \ \ \ 
\begin{aligned}
    \bar{e}_{i}^{\prime} &= \rho^{e \rightarrow v}(E_i^{\prime}) \\
    \bar{e}^{\prime} &= \rho^{e \rightarrow u}(E^{\prime}) \\
    \bar{v}^{\prime} &= \rho^{v \rightarrow u}(V^{\prime})
\end{aligned}
```

where ``\phi`` and ``\rho`` denote update functions and aggregate functions, respectively. ``v_i`` and ``v_j`` are node features from node ``i`` and its neighbor node ``j``, ``e_{ij}`` is edge feature for edge ``(i, j)``, and ``u`` is global feature for whole graph. ``e_{ij}^{\prime}``, ``v_{i}^{\prime}`` and ``u^{\prime}`` are new features for edge, node and global. ``\bar{e}_{i}^{\prime}``, ``\bar{e}^{\prime}`` and ``\bar{v}^{\prime}`` are aggregated new features for edge, node and global.

> Reference: [Battaglia2018](@cite)

Ingraph network, it describes an update order: edge, node and then global. There are three corresponding update functions for edge, node and then global, respectively. Three update functions return their default values as follow:

```
update_edge(gn, e, vi, vj, u) = e
update_vertex(gn, ē, vi, u) = vi
update_global(gn, ē, v̄, u) = u
```

Information propagation between different levels are achieved by aggregate functions. Three aggregate functions `aggregate_neighbors`, `aggregate_edges` and `aggregate_vertices` are defined to aggregate states.

GN block is realized into a abstract type `GraphNet`. User can make a subtype of `GraphNet` to customize GN block. Thus, a GN block is defined as a layer in GNN. `MessagePassing` is a subtype of `GraphNet`.

## Update functions

`update_edge` acts as the first update function to apply to edge states. It takes edge state `e`, node `i` state `vi`, node `j` state `vj` and global state `u`. It is expected to return a feature vector for new edge state. `update_vertex` updates nodes state by taking aggregated edge state `ē`, node `i` state `vi` and global state `u`. It is expected to return a feature vector for new node state. `update_global` updates global state with aggregated information from edge and node. It takes aggregated edge state `ē`, aggregated node state `v̄` and global state `u`. It is expected to return a feature vector for new global state. User can define their own behavior by overriding update functions.

```@docs
GeometricFlux.update_edge
GeometricFlux.update_vertex
GeometricFlux.update_global
GeometricFlux.update_batch_edge
GeometricFlux.update_batch_vertex
```

## Aggregate functions

An aggregate function `aggregate_neighbors` aggregates edge states for edges incident to some node `i` into node-level information. Aggregate function `aggregate_edges` aggregates all edge states into global-level information. The last aggregate function `aggregate_vertices` aggregates all vertex states into global-level information.

```@docs
GeometricFlux.aggregate_neighbors
GeometricFlux.aggregate_edges
GeometricFlux.aggregate_vertices
```

It is available for assigning aggregate function by assigning aggregate operations to `propagate` function.

```
propagate(gn, fg::FeaturedGraph, naggr=nothing, eaggr=nothing, vaggr=nothing)
```

`naggr`, `eaggr` and `vaggr` are arguments for `aggregate_neighbors`, `aggregate_edges` and `aggregate_vertices`, respectively. Available aggregate functions are assigned by following symbols to them: `:add`, `:sub`, `:mul`, `:div`, `:max`, `:min` and `:mean`.
