"""
    EEquivGraphConv(in_dim=>out_dim, pos_dim, edge_dim; init=glorot_uniform)

E(n)-equivariant graph neural network layer.

# Arguments

- `in_dim::Int`: The dimension of node feature. Data is assumed to be the form of `[feature; coordinate]`,
    so `in_dim` must strictly be less than the dimension of the input vectors.
- `out_dim`: The output of the layer will have dimension `out_dim` + (dimension of input vector - `in_dim`).
- `pos_dim::Int`: The dimension of positional encoding.
- `edge_dim::Int`: The dimension of edge feature.
- `init`: Weights' initialization function.

# Examples

```jldoctest
julia> in_dim, out_dim, pos_dim = 3, 5, 2
(3, 5, 2)

julia> egnn = EEquivGraphConv(in_dim=>out_dim, pos_dim, in_dim)
EEquivGraphConv(ϕ_edge=Chain(Dense(10 => 2), Dense(2 => 2)), ϕ_x=Chain(Dense(2 => 2), Dense(2 => 1; bias=false)), ϕ_h=Chain(Dense(5 => 2), Dense(2 => 5)))
```

See also [`WithGraph`](@ref) for training layer with static graph and [`EEquivGraphPE`](@ref) for positional encoding.
"""
struct EEquivGraphConv{X,E,H} <: AbstractGraphLayer
    pe::X
    nn_edge::E
    nn_h::H
end

@functor EEquivGraphConv

Flux.trainable(l::EEquivGraphConv) = (l.pe, l.nn_edge, l.nn_h)

function EEquivGraphConv(ch::Pair{Int,Int}, hidden_dim::Int, edge_dim::Int=0, σ=identity; init=glorot_uniform, use_tanh=false)
    in_dim, out_dim = ch
    nn_edge = Chain(
        Flux.Dense(2in_dim + edge_dim + 1, hidden_dim, σ; init=init),
        Flux.Dense(hidden_dim, hidden_dim, σ; init=init),
    )
    pe_σ = use_tanh ? tanh : identity
    pe = Chain(
        Flux.Dense(hidden_dim, hidden_dim, σ; init=init),
        Flux.Dense(hidden_dim, 1, pe_σ; bias=false, init=(x...)->init(x...; gain=0.001)),
    )
    nn_h = Chain(
        Flux.Dense(in_dim + hidden_dim, hidden_dim, σ; init=init),
        Flux.Dense(hidden_dim, out_dim; init=init)
    )
    return EEquivGraphConv(ch, hidden_dim, edge_dim, pe, nn_edge, nn_h)
end

function EEquivGraphConv(ch::Pair{Int,Int}, hidden_dim::Int, edge_dim::Int, pe, nn_edge, nn_h)
    in_dim, out_dim = ch
    @assert Flux.outputsize(pe, (hidden_dim, 2)) == (1, 2)
    @assert Flux.outputsize(nn_edge, (2in_dim + edge_dim + 1, 2)) == (hidden_dim, 2)
    @assert Flux.outputsize(nn_h, (in_dim + hidden_dim, 2)) == (out_dim, 2)
    return EEquivGraphConv(EEquivGraphPE(pe), nn_edge, nn_h)
end

ϕ_edge(l::EEquivGraphConv, h_i, h_j, dist, a) = l.nn_edge(vcat(h_i, h_j, dist, a))
ϕ_edge(l::EEquivGraphConv, h_i, h_j, dist, ::Nothing) = l.nn_edge(vcat(h_i, h_j, dist))

function message(l::EEquivGraphConv, h_i, h_j, x_i, x_j, e)
    dist = sum(abs2, x_i - x_j; dims=1)
    return ϕ_edge(l, h_i, h_j, dist, e)
end

update(l::EEquivGraphConv, m, h) = l.nn_h(vcat(h, m))

# For variable graph
function(egnn::EEquivGraphConv)(fg::AbstractFeaturedGraph)
    nf = node_feature(fg)
    ef = edge_feature(fg)
    pf = positional_feature(fg)
    GraphSignals.check_num_nodes(fg, nf)
    isnothing(ef) || GraphSignals.check_num_edges(fg, ef)
    _, V, X = propagate(egnn, graph(fg), ef, nf, pf, +)
    return ConcreteFeaturedGraph(fg, nf=V, pf=X)
end

# For static graph
function(l::EEquivGraphConv)(el::NamedTuple, X::AbstractArray, H::AbstractArray, E::AbstractArray)
    GraphSignals.check_num_nodes(el.N, H)
    GraphSignals.check_num_nodes(el.N, X)
    GraphSignals.check_num_edges(el.E, E)
    _, V, X = propagate(l, el, E, H, X, +)
    return V, X
end

function(l::EEquivGraphConv)(el::NamedTuple, X::AbstractArray, H::AbstractArray)
    GraphSignals.check_num_nodes(el.N, H)
    GraphSignals.check_num_nodes(el.N, X)
    _, V, X = propagate(l, el, nothing, H, X, +)
    return V, X
end

function Base.show(io::IO, l::EEquivGraphConv)
    print(io, "EEquivGraphConv(ϕ_edge=", l.nn_edge)
    print(io, ", ϕ_x=", l.pe.nn)
    print(io, ", ϕ_h=", l.nn_h)
    print(io, ")")
end

aggregate_neighbors(::EEquivGraphConv, el::NamedTuple, aggr, E) = scatter(aggr, E, el.xs, el.N)
aggregate_neighbors(::EEquivGraphConv, el::NamedTuple, aggr, E::AbstractMatrix) = scatter(aggr, E, el.xs)

aggregate_neighbors(::EEquivGraphConv, ::NamedTuple, ::Nothing, E) = nothing
aggregate_neighbors(::EEquivGraphConv, ::NamedTuple, ::Nothing, ::AbstractMatrix) = nothing

propagate(l::EEquivGraphConv, sg::SparseGraph, E, V, X, aggr) =
    propagate(l, GraphSignals.to_namedtuple(sg), E, V, X, aggr)

function propagate(l::EEquivGraphConv, el::NamedTuple, E, V, X, aggr)
    E = message(
        l, gather(V, el.xs), gather(V, el.nbrs),
        gather(X, el.xs), gather(X, el.nbrs),
        gather(E, el.es)
        )
    X = positional_encode(l.pe, el, X, E)
    Ē = aggregate_neighbors(l, el, aggr, E)
    V = update(l, Ē, V)
    return E, V, X
end

WithGraph(fg::AbstractFeaturedGraph, l::EEquivGraphConv) =
    WithGraph(GraphSignals.to_namedtuple(fg), l, positional_feature(fg))

WithGraph(fg::AbstractFeaturedGraph,
          l::EEquivGraphConv,
          pos::GraphSignals.AbstractGraphDomain) =
    WithGraph(GraphSignals.to_namedtuple(fg), l, pos)

(wg::WithGraph{<:EEquivGraphConv})(args...) = wg.layer(wg.graph, wg.position, args...)
