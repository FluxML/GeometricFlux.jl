"""
    GCNConv([graph,] in => out, σ=identity; bias=true, init=glorot_uniform)

Graph convolutional layer.

# Arguments

- `graph`: Optionally pass a FeaturedGraph.
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

@functor GCNConv

function (l::GCNConv)(fg::FeaturedGraph, x::AbstractMatrix)
    L̃ = normalized_laplacian(fg, eltype(x); selfloop=true)
    l.σ.(l.weight * x * L̃ .+ l.bias)
end

(l::GCNConv)(fg::FeaturedGraph) = FeaturedGraph(fg.graph, nf = l(fg, node_feature(fg)))
(l::GCNConv)(x::AbstractMatrix) = l(l.fg, x)

function Base.show(io::IO, l::GCNConv)
    out, in = size(l.weight)
    print(io, "GCNConv($in => $out")
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end


"""
    ChebConv([graph, ]in=>out, k; bias=true, init=glorot_uniform)

Chebyshev spectral graph convolutional layer.

# Arguments

- `graph`: Optionally pass a FeaturedGraph.
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

ChebConv(ch::Pair{Int,Int}, k::Int; kwargs...) =
    ChebConv(NullGraph(), ch, k; kwargs...)

@functor ChebConv

function (c::ChebConv)(fg::FeaturedGraph, X::AbstractMatrix{T}) where T
    L̃ = scaled_laplacian(fg, T)    
    
    @assert size(X, 1) == c.in_channel "Input feature size must match input channel size."
    GraphSignals.check_num_node(L̃, X)
    
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

(l::ChebConv)(fg::FeaturedGraph) = FeaturedGraph(fg.graph, nf = l(fg, node_feature(fg)))
(l::ChebConv)(x::AbstractMatrix) = l(l.fg, x)

function Base.show(io::IO, l::ChebConv)
    print(io, "ChebConv(", l.in_channel, " => ", l.out_channel)
    print(io, ", k=", l.k)
    print(io, ")")
end


"""
    GraphConv([graph,] in => out, σ=identity, aggr=+; bias=true, init=glorot_uniform)

Graph neural network layer.

# Arguments

- `graph`: Optionally pass a FeaturedGraph.
- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `aggr`: An aggregate function applied to the result of message function. `+`, `-`,
`*`, `/`, `max`, `min` and `mean` are available.
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

GraphConv(ch::Pair{Int,Int}, σ=identity, aggr=+; kwargs...) =
    GraphConv(NullGraph(), ch, σ, aggr; kwargs...)

@functor GraphConv

message(g::GraphConv, x_i, x_j::AbstractVector, e_ij) = g.weight2 * x_j

update(g::GraphConv, m::AbstractVector, x::AbstractVector) = g.σ.(g.weight1*x .+ m .+ g.bias)

function (gc::GraphConv)(fg::FeaturedGraph, x::AbstractMatrix)
    g = graph(fg)
    GraphSignals.check_num_node(g, x)
    _, x = propagate(gc, adjacency_list(g), Fill(0.f0, 0, ne(g)), x, +)
    x
end

(l::GraphConv)(fg::FeaturedGraph) = FeaturedGraph(fg.graph, nf = l(fg, node_feature(fg)))
(l::GraphConv)(x::AbstractMatrix) = l(l.fg, x)

function Base.show(io::IO, l::GraphConv)
    in_channel = size(l.weight1, ndims(l.weight1))
    out_channel = size(l.weight1, ndims(l.weight1)-1)
    print(io, "GraphConv(", in_channel, " => ", out_channel)
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

- `graph`: Optionally pass a FeaturedGraph.
- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `bias::Bool`: Keyword argument, whether to learn the additive bias.
- `heads`: Number attention heads 
- `concat`: Concatenate layer output or not. If not, layer output is averaged.
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

function (gat::GATConv)(fg::FeaturedGraph, X::AbstractMatrix)
    g = graph(fg)
    GraphSignals.check_num_node(g, X)
    _, X = propagate(gat, adjacency_list(g), Fill(0.f0, 0, ne(g)), X, +)
    X
end

(l::GATConv)(fg::FeaturedGraph) = FeaturedGraph(fg.graph, nf = l(fg, node_feature(fg)))
(l::GATConv)(x::AbstractMatrix) = l(l.fg, x)

function Base.show(io::IO, l::GATConv)
    in_channel = size(l.weight, ndims(l.weight))
    out_channel = size(l.weight, ndims(l.weight)-1)
    print(io, "GATConv(", in_channel, "=>", out_channel)
    print(io, ", LeakyReLU(λ=", l.negative_slope)
    print(io, "))")
end


"""
    GatedGraphConv([graph,] out, num_layers; aggr=+, init=glorot_uniform)

Gated graph convolution layer.

# Arguments

- `graph`: Optionally pass a FeaturedGraph.
- `out`: The dimension of output features.
- `num_layers`: The number of gated recurrent unit.
- `aggr`: An aggregate function applied to the result of message function. `+`, `-`,
`*`, `/`, `max`, `min` and `mean` are available.
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

GatedGraphConv(out_ch::Int, num_layers::Int; kwargs...) =
    GatedGraphConv(NullGraph(), out_ch, num_layers; kwargs...)

@functor GatedGraphConv

message(g::GatedGraphConv, x_i, x_j::AbstractVector, e_ij) = x_j

update(g::GatedGraphConv, m::AbstractVector, x) = m


function (ggc::GatedGraphConv)(fg::FeaturedGraph, H::AbstractMatrix{S}) where {T<:AbstractVector,S<:Real}
    adj = adjacency_list(fg)
    m, n = size(H)
    @assert (m <= ggc.out_ch) "number of input features must less or equals to output features."
    GraphSignals.check_num_node(adj, H)
    (m < ggc.out_ch) && (H = vcat(H, zeros(S, ggc.out_ch - m, n)))

    for i = 1:ggc.num_layers
        M = view(ggc.weight, :, :, i) * H
        _, M = propagate(ggc, adj, Fill(0.f0, 0, ne(adj)), M, +)
        H, _ = ggc.gru(H, M)  # BUG: FluxML/Flux.jl#1381
    end
    H
end

(l::GatedGraphConv)(fg::FeaturedGraph) = FeaturedGraph(fg.graph, nf = l(fg, node_feature(fg)))
(l::GatedGraphConv)(x::AbstractMatrix) = l(l.fg, x)


function Base.show(io::IO, l::GatedGraphConv)
    print(io, "GatedGraphConv(($(l.out_ch) => $(l.out_ch))^$(l.num_layers)")
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end



"""
    EdgeConv(graph, nn; aggr=max)

Edge convolutional layer.

# Arguments

- `graph`: Optionally pass a FeaturedGraph.
- `nn`: A neural network or a layer. It can be, e.g., MLP.
- `aggr`: An aggregate function applied to the result of message function. `+`, `-`,
`*`, `/`, `max`, `min` and `mean` are available.
"""
struct EdgeConv{V<:AbstractFeaturedGraph} <: MessagePassing
    fg::V
    nn
    aggr
end

EdgeConv(fg::AbstractFeaturedGraph, nn; aggr=max) = EdgeConv(fg, nn, aggr)
EdgeConv(nn; kwargs...) = EdgeConv(NullGraph(), nn; kwargs...)

@functor EdgeConv

message(e::EdgeConv, x_i::AbstractVector, x_j::AbstractVector, e_ij) = e.nn(vcat(x_i, x_j .- x_i))
update(e::EdgeConv, m::AbstractVector, x) = m

function (e::EdgeConv)(fg::FeaturedGraph, X::AbstractMatrix)
    g = graph(fg)
    GraphSignals.check_num_node(g, X)
    _, X = propagate(e, adjacency_list(g), Fill(0.f0, 0, ne(g)), X, e.aggr)
    X
end

(l::EdgeConv)(fg::FeaturedGraph) = FeaturedGraph(fg.graph, nf = l(fg, node_feature(fg)))
(l::EdgeConv)(x::AbstractMatrix) = l(l.fg, x)

function Base.show(io::IO, l::EdgeConv)
    print(io, "EdgeConv(", l.nn)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end


"""
    GINConv([fg,] nn, [eps])

    Graph Isomorphism Network.

# Arguments

- `fg`: Optionally pass in a FeaturedGraph as input.
- `nn`: A neural network/layer.
- `eps`: Weighting factor. Default 0.

The definition of this is as defined in the original paper,
Xu et. al. (2018) https://arxiv.org/abs/1810.00826.
"""

struct GINConv{V<:AbstractFeaturedGraph,R<:Real} <: MessagePassing
    fg::V
    nn
    eps::R
end

function GINConv(fg::AbstractFeaturedGraph, nn; eps=0f0)
    GINConv(fg, nn, eps)
end

function GINConv(nn; eps=0f0) 
    GINConv(NullGraph(), nn, eps)
end

Flux.trainable(g::GINConv) = (fg=g.fg,nn=g.nn)

message(g::GINConv, x_i::AbstractVector, x_j::AbstractVector) = x_j 
update(g::GINConv, m::AbstractVector, x) = g.nn((1 + g.eps) * x + m)

@functor GINConv

function (g::GINConv)(fg::FeaturedGraph, X::AbstractMatrix)
    gf = graph(fg)
    GraphSignals.check_num_node(gf, X)
    _, X = propagate(g, adjacency_list(gf), Fill(0.f0, 0, ne(gf)), X, +)
    X
end

(l::GINConv)(x::AbstractMatrix) = l(l.fg, x)
(l::GINConv)(fg::FeaturedGraph) = FeaturedGraph(fg.graph, nf = l(fg, node_feature(fg)))
