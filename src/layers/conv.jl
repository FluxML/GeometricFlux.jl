const AGGR2STR = Dict{Symbol,String}(:add => "∑", :sub => "-∑", :mul => "∏", :div => "1/∏",
                                     :max => "max", :min => "min", :mean => "𝔼[]")

"""
    GCNConv([graph, ]in=>out)
    GCNConv([graph, ]in=>out, σ)

Graph convolutional layer.

# Arguments
- `graph`: should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs)
or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs). Is optionnal so you can give a `FeaturedGraph` to
the layer instead of only the features.
- `in`: the dimension of input features.
- `out`: the dimension of output features.
- `bias::Bool=true`: keyword argument, whether to learn the additive bias.

Data should be stored in (# features, # nodes) order.
For example, a 1000-node graph each node of which poses 100 features is constructed.
The input data would be a `1000×100` array.
"""
struct GCNConv{T,F,S<:AbstractFeaturedGraph}
    weight::AbstractMatrix{T}
    bias::AbstractVector{T}
    σ::F
    fg::S
end

function GCNConv(ch::Pair{<:Integer,<:Integer}, σ = identity;
                 init=glorot_uniform, T::DataType=Float32, bias::Bool=true)
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    fg = NullGraph()
    GCNConv(T.(init(ch[2], ch[1])), b, σ, fg)
end

function GCNConv(adj::AbstractMatrix, ch::Pair{<:Integer,<:Integer}, σ = identity;
                 init=glorot_uniform, T::DataType=Float32, bias::Bool=true)
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    fg = FeaturedGraph(adj)
    GCNConv(T.(init(ch[2], ch[1])), b, σ, fg)
end

function GCNConv(fg::FeaturedGraph, ch::Pair{<:Integer,<:Integer}, σ = identity;
                 init=glorot_uniform, T::DataType=Float32, bias::Bool=true)
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    GCNConv(T.(init(ch[2], ch[1])), b, σ, fg)
end

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
    ChebConv([graph, ]in=>out, k)

Chebyshev spectral graph convolutional layer.

# Arguments
- `graph`: should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`,
`SimpleWeightedDiGraph` (from SimpleWeightedGraphs). Is optionnal so you can give a `FeaturedGraph` to
the layer instead of only the features.
- `in`: the dimension of input features.
- `out`: the dimension of output features.
- `k`: the order of Chebyshev polynomial.
- `bias::Bool=true`: keyword argument, whether to learn the additive bias.
"""
struct ChebConv{T,S<:AbstractFeaturedGraph}
    weight::AbstractArray{T,3}
    bias::AbstractVector{T}
    fg::S
    k::Integer
    in_channel::Integer
    out_channel::Integer
end

function ChebConv(adj::AbstractMatrix, ch::Pair{<:Integer,<:Integer}, k::Integer;
                  init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    b = bias ? init(ch[2]) : zeros(T, ch[2])
    fg = FeaturedGraph(adj)
    ChebConv(init(ch[2], ch[1], k), b, fg, k, ch[1], ch[2])
end

function ChebConv(ch::Pair{<:Integer,<:Integer}, k::Integer;
                  init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    b = bias ? init(ch[2]) : zeros(T, ch[2])
    fg = NullGraph()
    ChebConv(init(ch[2], ch[1], k), b, fg, k, ch[1], ch[2])
end

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
    GraphConv([graph, ]in=>out)
    GraphConv([graph, ]in=>out, σ)
    GraphConv([graph, ]in=>out, σ, aggr)

Graph neural network layer.

# Arguments
- `graph`: should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`,
`SimpleWeightedDiGraph` (from SimpleWeightedGraphs). Is optionnal so you can give a `FeaturedGraph` to
the layer instead of only the features.
- `in`: the dimension of input features.
- `out`: the dimension of output features.
- `bias::Bool=true`: keyword argument, whether to learn the additive bias.
- `σ=identity`: activation function.
- `aggr::Symbol=:add`: an aggregate function applied to the result of message function. `:add`, `:max` and `:mean` are available.
"""
struct GraphConv{V<:AbstractFeaturedGraph,T} <: MessagePassing
    fg::V
    weight1::AbstractMatrix{T}
    weight2::AbstractMatrix{T}
    bias::AbstractVector{T}
    σ
    aggr::Symbol
end

function GraphConv(el::AbstractVector{<:AbstractVector{<:Integer}},
                   ch::Pair{<:Integer,<:Integer}, σ=identity, aggr=:add;
                   init = glorot_uniform, bias::Bool=true, T::DataType=Float32)
    w1 = T.(init(ch[2], ch[1]))
    w2 = T.(init(ch[2], ch[1]))
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    fg = FeaturedGraph(el)
    GraphConv(fg, w1, w2, b, σ, aggr)
end

function GraphConv(adj::AbstractMatrix, ch::Pair{<:Integer,<:Integer}, σ=identity, aggr=:add;
                   init = glorot_uniform, bias::Bool=true, T::DataType=Float32)
    w1 = T.(init(ch[2], ch[1]))
    w2 = T.(init(ch[2], ch[1]))
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    fg = FeaturedGraph(adjacency_list(adj))
    GraphConv(fg, w1, w2, b, σ, aggr)
end

function GraphConv(ch::Pair{<:Integer,<:Integer}, σ=identity, aggr=:add;
                   init = glorot_uniform, bias::Bool=true, T::DataType=Float32)
    w1 = T.(init(ch[2], ch[1]))
    w2 = T.(init(ch[2], ch[1]))
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    GraphConv(NullGraph(), w1, w2, b, σ, aggr)
end

@functor GraphConv

message(g::GraphConv, x_i, x_j::AbstractVector, e_ij) = g.weight2 * x_j
update(g::GraphConv, m::AbstractVector, x::AbstractVector) = g.σ.(g.weight1*x .+ m .+ g.bias)
function (gc::GraphConv)(X::AbstractMatrix)
    @assert has_graph(gc.fg) "A GraphConv created without a graph must be given a FeaturedGraph as an input."
    g = graph(gc.fg)
    Zygote.ignore() do
        GraphSignals.check_num_node(g, X)
    end
    _, X = propagate(gc, adjacency_list(g), Fill(0.f0, 0, ne(g)), X, :add)
    X
end
function (g::GraphConv)(fg::FeaturedGraph)
    Zygote.ignore() do
        GraphSignals.check_num_node(graph(fg), node_feature(fg))
    end
    propagate(g, fg, :add)
end

function Base.show(io::IO, l::GraphConv)
    in_channel = size(l.weight1, ndims(l.weight1))
    out_channel = size(l.weight1, ndims(l.weight1)-1)
    print(io, "GraphConv(G(V=", nv(l.fg), ", E=", ne(l.fg))
    print(io, "), ", in_channel, "=>", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", aggr=", AGGR2STR[l.aggr])
    print(io, ")")
end



"""
    GATConv([graph, ]in=>out)

Graph attentional layer.

# Arguments
- `graph`: should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`,
`SimpleWeightedDiGraph` (from SimpleWeightedGraphs). Is optionnal so you can give a `FeaturedGraph` to
the layer instead of only the features.
- `in`: the dimension of input features.
- `out`: the dimension of output features.
- `bias::Bool=true`: keyword argument, whether to learn the additive bias.
- `negative_slope::Real=0.2`: keyword argument, the parameter of LeakyReLU.
"""
struct GATConv{V<:AbstractFeaturedGraph,T<:Real} <: MessagePassing
    fg::V
    weight::AbstractMatrix{T}
    bias::AbstractVector{T}
    a::AbstractMatrix{T}
    negative_slope::T
    channel::Pair{<:Integer,<:Integer}
    heads::Integer
    concat::Bool
end

function GATConv(adj::AbstractMatrix, ch::Pair{<:Integer,<:Integer}; T::DataType=Float32,
                 heads::Integer=1, concat::Bool=true, negative_slope::Real=T(0.2),
                 init=glorot_uniform, bias::Bool=true)
    w = T.(init(ch[2]*heads, ch[1]))
    b = bias ? T.(init(ch[2]*heads)) : zeros(T, ch[2]*heads)
    a = T.(init(2*ch[2], heads))
    fg = FeaturedGraph(adjacency_list(adj))
    GATConv(fg, w, b, a, negative_slope, ch, heads, concat)
end

function GATConv(ch::Pair{<:Integer,<:Integer}; T::DataType=Float32,
                 heads::Integer=1, concat::Bool=true, negative_slope::Real=T(0.2),
                 init=glorot_uniform, bias::Bool=true)
    w = T.(init(ch[2]*heads, ch[1]))
    b = bias ? T.(init(ch[2]*heads)) : zeros(T, ch[2]*heads)
    a = T.(init(2*ch[2], heads))
    GATConv(NullGraph(), w, b, a, negative_slope, ch, heads, concat)
end

@functor GATConv

# Here the α that has not been softmaxed is the first number of the output message
function message(g::GATConv, x_i::AbstractVector, x_j::AbstractVector)
    x_i = reshape(g.weight*x_i, :, g.heads)
    x_j = reshape(g.weight*x_j, :, g.heads)
    n = size(x_i, 1)
    e = vcat(x_i, x_j+zero(x_j))
    e = sum(e .* g.a, dims=1)  # inner product for each head, output shape: (1, g.heads)
    e = leakyrelu.(e, g.negative_slope)
    vcat(e, x_j)  # shape: (n+1, g.heads)
end

# After some reshaping due to the multihead, we get the α from each message,
# then get the softmax over every α, and eventually multiply the message by α
function apply_batch_message(g::GATConv, i, js, X::AbstractMatrix)
    e_ij = hcat([message(g, get_feature(X, i), get_feature(X, j)) for j = js]...)
    n = size(e_ij, 1)
    alphas = Flux.softmax(reshape(view(e_ij, 1, :), g.heads, :), dims=2)
    msgs = view(e_ij, 2:n, :) .* reshape(alphas, 1, :)
    reshape(msgs, (n-1)*g.heads, :)
end

update_batch_edge(g::GATConv, adj, E::AbstractMatrix, X::AbstractMatrix, u) = update_batch_edge(g, adj, X)

function update_batch_edge(g::GATConv, adj, X::AbstractMatrix)
    n = size(adj, 1)
    # a vertex must always receive a message from itself
    Zygote.ignore() do
        add_self_loop!(adj, n)
    end
    hcat([apply_batch_message(g, i, adj[i], X) for i in 1:n]...)
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
    _, X = propagate(gat, adjacency_list(g), Fill(0.f0, 0, ne(g)), X, :add)
    X
end
function (g::GATConv)(fg::FeaturedGraph)
    Zygote.ignore() do
        GraphSignals.check_num_node(graph(fg), node_feature(fg))
    end
    propagate(g, fg, :add)
end

function Base.show(io::IO, l::GATConv)
    in_channel = size(l.weight, ndims(l.weight))
    out_channel = size(l.weight, ndims(l.weight)-1)
    print(io, "GATConv(G(V=", nv(l.fg), ", E=", ne(l.fg))
    print(io, "), ", in_channel, "=>", out_channel)
    print(io, ", LeakyReLU(λ=", l.negative_slope)
    print(io, "))")
end



"""
    GatedGraphConv([graph, ]out, num_layers)

Gated graph convolution layer.

# Arguments
- `graph`: should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`,
`SimpleWeightedDiGraph` (from SimpleWeightedGraphs). Is optionnal so you can give a `FeaturedGraph` to
the layer instead of only the features.
- `out`: the dimension of output features.
- `num_layers` specifies the number of gated recurrent unit.
- `aggr::Symbol=:add`: an aggregate function applied to the result of message function. `:add`, `:max` and `:mean` are available.
"""
struct GatedGraphConv{V<:AbstractFeaturedGraph, T <: Real, R} <: MessagePassing
    fg::V
    weight::AbstractArray{T}
    gru::R
    out_ch::Integer
    num_layers::Integer
    aggr::Symbol
end

function GatedGraphConv(adj::AbstractMatrix, out_ch::Integer, num_layers::Integer;
                        aggr=:add, init=glorot_uniform, T::DataType=Float32)
    w = T.(init(out_ch, out_ch, num_layers))
    gru = GRUCell(out_ch, out_ch)
    fg = FeaturedGraph(adjacency_list(adj))
    GatedGraphConv(fg, w, gru, out_ch, num_layers, aggr)
end

function GatedGraphConv(out_ch::Integer, num_layers::Integer;
                        aggr=:add, init=glorot_uniform, T::DataType=Float32)
    w = T.(init(out_ch, out_ch, num_layers))
    gru = GRUCell(out_ch, out_ch)
    GatedGraphConv(NullGraph(), w, gru, out_ch, num_layers, aggr)
end

@functor GatedGraphConv

message(g::GatedGraphConv, x_i, x_j::AbstractVector, e_ij) = x_j
update(g::GatedGraphConv, m::AbstractVector, x) = m

function (ggc::GatedGraphConv)(X::AbstractMatrix{T}) where {T<:Real}
    @assert has_graph(ggc.fg) "A GraphConv created without a graph must be given a FeaturedGraph as an input."
    ggc(adjacency_list(ggc.fg), X)
end

function (ggc::GatedGraphConv{V,T})(fg::FeaturedGraph) where {V,T<:Real}
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
        _, M = propagate(ggc, adj, Fill(0.f0, 0, ne(adj)), M, :add)
        H, _ = ggc.gru(H, M)  # BUG: FluxML/Flux.jl#1381
    end
    H
end

function Base.show(io::IO, l::GatedGraphConv)
    print(io, "GatedGraphConv(G(V=", nv(l.fg), ", E=", ne(l.fg))
    print(io, "), (=>", l.out_ch)
    print(io, ")^", l.num_layers)
    print(io, ", aggr=", AGGR2STR[l.aggr])
    print(io, ")")
end



"""
    EdgeConv(graph, nn)
    EdgeConv(graph, nn, aggr)

Edge convolutional layer.

# Arguments
- `graph`: should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `nn`: a neural network
- `aggr::Symbol=:max`: an aggregate function applied to the result of message function. `:add`, `:max` and `:mean` are available.
"""
struct EdgeConv{V<:AbstractFeaturedGraph} <: MessagePassing
    fg::V
    nn
    aggr::Symbol
end

function EdgeConv(adj::AbstractMatrix, nn; aggr::Symbol=:max)
    fg = FeaturedGraph(adjacency_list(adj))
    EdgeConv(fg, nn, aggr)
end

function EdgeConv(nn; aggr::Symbol=:max)
    EdgeConv(NullGraph(), nn, aggr)
end

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

function (e::EdgeConv)(fg::FeaturedGraph)
    Zygote.ignore() do
        GraphSignals.check_num_node(graph(fg), node_feature(fg))
    end
    propagate(e, fg, e.aggr)
end

function Base.show(io::IO, l::EdgeConv)
    print(io, "EdgeConv(G(V=", nv(l.fg), ", E=", ne(l.fg))
    print(io, "), ", l.nn)
    print(io, ", aggr=", AGGR2STR[l.aggr])
    print(io, ")")
end


"""
    GINConv([graph, ] nn, eps, train_eps
"""

struct GINConv{V<:AbstractFeaturedGraph,R<:Real} <: MessagePassing
    fg::V
    nn
    eps::Untrainable{R}
end

function GINConv(fg::V, nn, eps=zero(T)) where {V <: AbstractFeaturedGraph, 
                                                T <: Real}
    GINConv(fg, nn, Untrainable(eps))
end

function GINConv(nn, eps=zero(T)) where {T <: Real}
    GINConv(NullGraph(), nn, Untrainable(eps))
end

message(g::GINConv, x_i::AbstractVector, x_j::AbstractVector) = x_j 
update(g::GINConv, m::AbstractVector, x) = g.nn((1.0 + g.eps) * x + m)

@functor GINConv

function(g::GINConv)(X::AbstractMatrix)
    fg = FeaturedGraph(graph(g.fg), nf=X)
    Zygote.ignore() do
        GraphSignals.check_num_node(graph(g.fg), X)
    end
    propagate(g, fg, :add)
end

function(g::GINConv)(fg::FeaturedGraph)
    Zygote.ignore() do
        GraphSignals.check_num_node(graph(fg), node_feature(fg))
    end
    propagate(g, fg, :add)
end
