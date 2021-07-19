"""
    GCNConv([graph,] in => out, σ=identity; bias=true, init=glorot_uniform)

Graph convolutional layer.

# Arguments

- `graph`: Should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs). Is optionnal so you can give a `FeaturedGraph` to
the layer instead of only the features.
- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.


The input to the layer is a node feature array `X` 
of size `(num_features, num_nodes)`.
"""
struct GCNConv{A<:AbstractMatrix, B, F, S<:AbstractFeaturedGraph}
    weight::A
    bias::B
    σ::F
    fg::S
end

function GCNConv(fg::AbstractFeaturedGraph, ch::Pair{Int,Int}, σ=identity;
                 init=glorot_uniform, bias::Bool=true)
    in, out = ch
    W = init(out, in)
    b = Flux.create_bias(W, bias, out)
    GCNConv(W, b, σ, fg)
end

GCNConv(ch::Pair{Int,Int}, σ = identity; kwargs...) =
    GCNConv(NullGraph(), ch, σ; kwargs...)

GCNConv(adj::AbstractMatrix, ch::Pair{Int,Int}, σ = identity; kwargs...) =
    GCNConv(FeaturedGraph(adj), ch, σ; kwargs...)

@functor GCNConv

function (g::GCNConv)(L̃::T, X::T) where {T<:AbstractMatrix}
    Zygote.ignore() do
        GraphSignals.check_num_node(L̃, X)
    end
    g.σ.(g.weight * X * L̃ .+ g.bias)
end
(g::GCNConv)(L̃::AbstractMatrix, X::Transpose{T,R}) where {T<:Real,R<:AbstractMatrix} = g(L̃, R(X))

function (g::GCNConv)(X::AbstractMatrix{T}) where {T}
    @assert has_graph(g.fg) "A GCNConv created without a graph must be given a FeaturedGraph as an input."
    A = adjacency_matrix(g.fg)
    L̃ = normalized_laplacian(A, eltype(X); selfloop=true)
    g(L̃, X)
end

function (g::GCNConv)(fg::FeaturedGraph)
    X = node_feature(fg)
    A = adjacency_matrix(fg) # TODO: choose graph from g or fg
    Zygote.ignore() do
        g.fg isa NullGraph || (g.fg.graph = A)
    end
    L̃ = normalized_laplacian(A, eltype(X); selfloop=true)
    X_ = g(L̃, X)
    FeaturedGraph(A, nf=X_)
end

function Base.show(io::IO, l::GCNConv)
    in_channel = size(l.weight, ndims(l.weight))
    out_channel = size(l.weight, ndims(l.weight)-1)
    print(io, "GCNConv(G(V=", nv(l.fg))
    print(io, ", E), ", in_channel, "=>", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end



"""
    ChebConv([graph, ]in=>out, k; bias=true, init=glorot_uniform)

Chebyshev spectral graph convolutional layer.

# Arguments

- `graph`: Should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs). Is optionnal so you can give a `FeaturedGraph` to
the layer instead of only the features.
- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `k`: The order of Chebyshev polynomial.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.
"""
struct ChebConv{A<:AbstractArray{<:Number,3}, B, S<:AbstractFeaturedGraph}
    weight::A
    bias::B
    fg::S
    k::Int
    in_channel::Int
    out_channel::Int
end

function ChebConv(fg::AbstractFeaturedGraph, ch::Pair{Int,Int}, k::Int;
                  init=glorot_uniform, bias::Bool=true)
    in, out = ch
    W = init(out, in, k)
    b = Flux.create_bias(W, bias, out)
    ChebConv(W, b, fg, k, in, out)
end

ChebConv(adj::AbstractMatrix, ch::Pair{Int,Int}, k::Int; kwargs...) =
    ChebConv(FeaturedGraph(adj), ch, k; kwargs...)

ChebConv(ch::Pair{Int,Int}, k::Int; kwargs...) =
    ChebConv(NullGraph(), ch, k; kwargs...)

@functor ChebConv

function (c::ChebConv)(L̃::T, X::T) where {T<:AbstractMatrix}
    @assert size(X, 1) == c.in_channel "Input feature size must match input channel size."
    Zygote.ignore() do
        GraphSignals.check_num_node(L̃, X)
    end

    Z_prev = X
    Z = X * L̃
    Y = view(c.weight,:,:,1) * Z_prev
    Y += view(c.weight,:,:,2) * Z
    for k = 3:c.k
        Z, Z_prev = 2*Z*L̃ - Z_prev, Z
        Y += view(c.weight,:,:,k) * Z
    end
    return Y .+ c.bias
end

(c::ChebConv)(L̃::AbstractMatrix, X::Transpose{T,R}) where {T<:Real,R<:AbstractMatrix} = c(L̃, R(X))

function (c::ChebConv)(X::AbstractMatrix{T}) where {T<:Real}
    @assert has_graph(c.fg) "A ChebConv created without a graph must be given a FeaturedGraph as an input."
    g = graph(c.fg)
    L̃ = scaled_laplacian(g, T)
    c(L̃, X)
end

function (c::ChebConv)(fg::FeaturedGraph)
    @assert has_graph(fg) "A given FeaturedGraph must contain a graph."
    g = graph(fg)
    Zygote.ignore() do
        c.fg isa NullGraph || (c.fg.graph = g)
    end
    X = node_feature(fg)
    L̃ = scaled_laplacian(adjacency_matrix(fg))
    X_ = c(L̃, X)
    FeaturedGraph(g, nf=X_)
end

function Base.show(io::IO, l::ChebConv)
    print(io, "ChebConv(G(V=", nv(l.fg))
    print(io, ", E), ", l.in_channel, "=>", l.out_channel)
    print(io, ", k=", l.k)
    print(io, ")")
end



"""
    GraphConv([graph,] in => out, σ=identity, aggr=+; bias=true, init=glorot_uniform)

Graph neural network layer.

# Arguments

- `graph`: Should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs). Is optionnal so you can give a `FeaturedGraph` to
the layer instead of only the features.
- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `aggr`: An aggregate function applied to the result of message function. `+`, `max` and `mean` are available.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.
"""
struct GraphConv{V<:AbstractFeaturedGraph, A<:AbstractMatrix, B} <: MessagePassing
    fg::V
    weight1::A
    weight2::A
    bias::B
    σ
    aggr
end

function GraphConv(fg::AbstractFeaturedGraph, ch::Pair{Int,Int}, σ=identity, aggr=+;
                   init=glorot_uniform, bias::Bool=true)
    in, out = ch
    W1 = init(out, in)
    W2 = init(out, in)
    b = Flux.create_bias(W1, bias, out)
    GraphConv(fg, W1, W2, b, σ, aggr)
end

GraphConv(el::AbstractVector{<:AbstractVector}, ch::Pair{Int,Int}, σ=identity, aggr=+; kwargs...) =
    GraphConv(FeaturedGraph(el), ch, σ, aggr; kwargs...)

GraphConv(adj::AbstractMatrix, ch::Pair{Int,Int}, σ=identity, aggr=+; kwargs...) =
    GraphConv(adjacency_list(adj), ch, σ, aggr; kwargs...)

GraphConv(ch::Pair{Int,Int}, σ=identity, aggr=+; kwargs...) =
    GraphConv(NullGraph(), ch, σ, aggr; kwargs...)

@functor GraphConv

message(g::GraphConv, x_i, x_j::AbstractVector, e_ij) = g.weight2 * x_j
update(g::GraphConv, m::AbstractVector, x::AbstractVector) = g.σ.(g.weight1*x .+ m .+ g.bias)
function (gc::GraphConv)(X::AbstractMatrix)
    @assert has_graph(gc.fg) "A GraphConv created without a graph must be given a FeaturedGraph as an input."
    g = graph(gc.fg)
    Zygote.ignore() do
        GraphSignals.check_num_node(g, X)
    end
    _, X = propagate(gc, adjacency_list(g), Fill(0.f0, 0, ne(g)), X, +)
    X
end

(g::GraphConv)(fg::FeaturedGraph) = propagate(g, fg, +)

function Base.show(io::IO, l::GraphConv)
    in_channel = size(l.weight1, ndims(l.weight1))
    out_channel = size(l.weight1, ndims(l.weight1)-1)
    print(io, "GraphConv(G(V=", nv(l.fg), ", E=", ne(l.fg))
    print(io, "), ", in_channel, "=>", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end



"""
    GATConv([graph,] in => out;
            heads=1,
            concat=true,
            init=glorot_uniform    
            bias=true, 
            negative_slope=0.2)

Graph attentional layer.

# Arguments

- `graph`: Should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs). Is optionnal so you can give a `FeaturedGraph` to
the layer instead of only the features.
- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `bias::Bool`: Keyword argument, whether to learn the additive bias.
- `heads`: number attention heads 
- `concat`
- `negative_slope::Real`: Keyword argument, the parameter of LeakyReLU.
"""
struct GATConv{V<:AbstractFeaturedGraph, T, A<:AbstractMatrix{T}, B} <: MessagePassing
    fg::V
    weight::A
    bias::B
    a::A
    negative_slope::T
    channel::Pair{Int, Int}
    heads::Int
    concat::Bool
end

function GATConv(fg::AbstractFeaturedGraph, ch::Pair{Int,Int};
                 heads::Int=1, concat::Bool=true, negative_slope=0.2f0,
                 init=glorot_uniform, bias::Bool=true)
    in, out = ch             
    W = init(out*heads, in)
    b = Flux.create_bias(W, bias, out*heads)
    a = init(2*out, heads)
    GATConv(fg, W, b, a, negative_slope, ch, heads, concat)
end

GATConv(el::AbstractVector{<:AbstractVector}, ch::Pair{Int,Int}; kwargs...) =
    GATConv(FeaturedGraph(el), ch; kwargs...)

GATConv(adj::AbstractMatrix, ch::Pair{Int,Int}; kwargs...) =
    GATConv(adjacency_list(adj), ch; kwargs...)

GATConv(ch::Pair{Int,Int}; kwargs...) = GATConv(NullGraph(), ch; kwargs...)

@functor GATConv

# Here the α that has not been softmaxed is the first number of the output message
function message(g::GATConv, x_i::AbstractVector, x_j::AbstractVector)
    x_i = reshape(g.weight*x_i, :, g.heads)
    x_j = reshape(g.weight*x_j, :, g.heads)
    x_ij = vcat(x_i, x_j+zero(x_j))
    e = sum(x_ij .* g.a, dims=1)  # inner product for each head, output shape: (1, g.heads)
    e_ij = leakyrelu.(e, g.negative_slope)
    vcat(e_ij, x_j)  # shape: (n+1, g.heads)
end

# After some reshaping due to the multihead, we get the α from each message,
# then get the softmax over every α, and eventually multiply the message by α
function apply_batch_message(g::GATConv, i, js, X::AbstractMatrix)
    e_ij = mapreduce(j -> GeometricFlux.message(g, _view(X, i), _view(X, j)), hcat, js)
    n = size(e_ij, 1)
    αs = Flux.softmax(reshape(view(e_ij, 1, :), g.heads, :), dims=2)
    msgs = view(e_ij, 2:n, :) .* reshape(αs, 1, :)
    reshape(msgs, (n-1)*g.heads, :)
end

update_batch_edge(g::GATConv, adj, E::AbstractMatrix, X::AbstractMatrix, u) = update_batch_edge(g, adj, X)

function update_batch_edge(g::GATConv, adj, X::AbstractMatrix)
    n = size(adj, 1)
    # a vertex must always receive a message from itself
    Zygote.ignore() do
        GraphLaplacians.add_self_loop!(adj, n)
    end
    mapreduce(i -> apply_batch_message(g, i, adj[i], X), hcat, 1:n)
end

# The same as update function in batch manner
update_batch_vertex(g::GATConv, M::AbstractMatrix, X::AbstractMatrix, u) = update_batch_vertex(g, M)

function update_batch_vertex(g::GATConv, M::AbstractMatrix)
    M = M .+ g.bias
    if !g.concat
        N = size(M, 2)
        M = reshape(mean(reshape(M, :, g.heads, N), dims=2), :, N)
    end
    return M
end

function (gat::GATConv)(X::AbstractMatrix)
    @assert has_graph(gat.fg) "A GATConv created without a graph must be given a FeaturedGraph as an input."
    g = graph(gat.fg)
    Zygote.ignore() do
        GraphSignals.check_num_node(g, X)
    end
    _, X = propagate(gat, adjacency_list(g), Fill(0.f0, 0, ne(g)), X, +)
    X
end

(g::GATConv)(fg::FeaturedGraph) = propagate(g, fg, +)

function Base.show(io::IO, l::GATConv)
    in_channel = size(l.weight, ndims(l.weight))
    out_channel = size(l.weight, ndims(l.weight)-1)
    print(io, "GATConv(G(V=", nv(l.fg), ", E=", ne(l.fg))
    print(io, "), ", in_channel, "=>", out_channel)
    print(io, ", LeakyReLU(λ=", l.negative_slope)
    print(io, "))")
end



"""
    GatedGraphConv([graph,] out, num_layers; aggr=+, init=glorot_uniform)

Gated graph convolution layer.

# Arguments

- `graph`: Should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs). Is optionnal so you can give a `FeaturedGraph` to
the layer instead of only the features.
- `out`: The dimension of output features.
- `num_layers`: The number of gated recurrent unit.
- `aggr`: Keyword argument, an aggregate function applied to the result of message function. `+`, `max` and `mean` are available.
"""
struct GatedGraphConv{V<:AbstractFeaturedGraph, A<:AbstractArray{<:Number,3}, R} <: MessagePassing
    fg::V
    weight::A
    gru::R
    out_ch::Int
    num_layers::Int
    aggr
end

function GatedGraphConv(fg::AbstractFeaturedGraph, out_ch::Int, num_layers::Int;
                        aggr=+, init=glorot_uniform)
    w = init(out_ch, out_ch, num_layers)
    gru = GRUCell(out_ch, out_ch)
    GatedGraphConv(fg, w, gru, out_ch, num_layers, aggr)
end

GatedGraphConv(el::AbstractVector{<:AbstractVector}, out_ch::Int, num_layers::Int; kwargs...) =
    GatedGraphConv(FeaturedGraph(el), out_ch, num_layers; kwargs...)

GatedGraphConv(adj::AbstractMatrix, out_ch::Int, num_layers::Int; kwargs...) =
    GatedGraphConv(adjacency_list(adj), out_ch, num_layers; kwargs...)

GatedGraphConv(out_ch::Int, num_layers::Int; kwargs...) =
    GatedGraphConv(NullGraph(), out_ch, num_layers; kwargs...)

@functor GatedGraphConv

message(g::GatedGraphConv, x_i, x_j::AbstractVector, e_ij) = x_j
update(g::GatedGraphConv, m::AbstractVector, x) = m

function (ggc::GatedGraphConv)(X::AbstractMatrix)
    @assert has_graph(ggc.fg) "A GraphConv created without a graph must be given a FeaturedGraph as an input."
    ggc(adjacency_list(ggc.fg), X)
end

function (ggc::GatedGraphConv)(fg::FeaturedGraph)
    g = graph(fg)
    H = ggc(adjacency_list(g), node_feature(fg))
    FeaturedGraph(g, nf=H)
end

function (ggc::GatedGraphConv)(adj::AbstractVector{T}, H::AbstractMatrix{S}) where {T<:AbstractVector,S<:Real}
    m, n = size(H)
    @assert (m <= ggc.out_ch) "number of input features must less or equals to output features."
    Zygote.ignore() do
        GraphSignals.check_num_node(adj, H)
    end
    (m < ggc.out_ch) && (H = vcat(H, zeros(S, ggc.out_ch - m, n)))

    for i = 1:ggc.num_layers
        M = view(ggc.weight, :, :, i) * H
        _, M = propagate(ggc, adj, Fill(0.f0, 0, ne(adj)), M, +)
        H, _ = ggc.gru(H, M)  # BUG: FluxML/Flux.jl#1381
    end
    H
end

function Base.show(io::IO, l::GatedGraphConv)
    print(io, "GatedGraphConv(G(V=", nv(l.fg), ", E=", ne(l.fg))
    print(io, "), (=>", l.out_ch)
    print(io, ")^", l.num_layers)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end



"""
    EdgeConv(graph, nn; aggr=max)

Edge convolutional layer.

# Arguments

- `graph`: Should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `nn`: A neural network or a layer. It can be, e.g., MLP.
- `aggr`: Keyword argument, an aggregate function applied to the result of message function. `+`, `max` and `mean` are available.
"""
struct EdgeConv{V<:AbstractFeaturedGraph} <: MessagePassing
    fg::V
    nn
    aggr
end

EdgeConv(fg::AbstractFeaturedGraph, nn; aggr=max) = EdgeConv(fg, nn, aggr)
EdgeConv(el::AbstractVector{<:AbstractVector}, nn; kwargs...) = EdgeConv(FeaturedGraph(el), nn; kwargs...)
EdgeConv(adj::AbstractMatrix, nn; kwargs...) = EdgeConv(adjacency_list(adj), nn; kwargs...)
EdgeConv(nn; kwargs...) = EdgeConv(NullGraph(), nn; kwargs...)

@functor EdgeConv

message(e::EdgeConv, x_i::AbstractVector, x_j::AbstractVector, e_ij) = e.nn(vcat(x_i, x_j .- x_i))
update(e::EdgeConv, m::AbstractVector, x) = m

function (e::EdgeConv)(X::AbstractMatrix)
    @assert has_graph(e.fg) "A EdgeConv created without a graph must be given a FeaturedGraph as an input."
    g = graph(e.fg)
    Zygote.ignore() do
        GraphSignals.check_num_node(g, X)
    end
    _, X = propagate(e, adjacency_list(g), Fill(0.f0, 0, ne(g)), X, e.aggr)
    X
end

(e::EdgeConv)(fg::FeaturedGraph) = propagate(e, fg, e.aggr)

function Base.show(io::IO, l::EdgeConv)
    print(io, "EdgeConv(G(V=", nv(l.fg), ", E=", ne(l.fg))
    print(io, "), ", l.nn)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end
