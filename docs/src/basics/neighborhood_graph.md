# Neighborhood Graphs

In machine learning, it is often that using a neighborhood graph to approach manifold in high dimensional space. The construction of neighborhood graph is the essential step for machine learning algorithms on graph/manifold, especially manifold learning.

The k-nearest neighbor (kNN) method is the most frequent use to construct a neighborhood graph. We provide [`kneighbors_graph`](@ref) to generate a kNN graph from a set of nodes/points.

We prepare 1,024 10-dimensional data points.

```julia
X = rand(Float32, 10, 1024)
```

Then, we can generate a kNN graph with `k=7`, which means a data point should be linked to their top-7 nearest neighbor points.

```julia
fg = kneighbors_graph(nf, 7)
```

The default distance metric would be `Euclidean` distance from Distance.jl package. If one wants to customize [`kneighbors_graph`](@ref) by using different distance metric, you can just use the distance objects from Distance.jl package directly, and pass it to [`kneighbors_graph`](@ref).

```julia
using Distances

fg = kneighbors_graph(nf, 7, Cityblock())
```
