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
                 init=glorot_uniform, T::DataType=Float32, bias::Bool=true, cache::Bool=true)
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    fg = cache ? FeaturedGraph() : NullGraph()
    GCNConv(T.(init(ch[2], ch[1])), b, σ, fg)
end

function GCNConv(adj::AbstractMatrix, ch::Pair{<:Integer,<:Integer}, σ = identity;
                 init=glorot_uniform, T::DataType=Float32, bias::Bool=true, cache::Bool=true)
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    fg = cache ? FeaturedGraph(adj) : NullGraph()
    GCNConv(T.(init(ch[2], ch[1])), b, σ, fg)
end

@functor GCNConv

function (g::GCNConv)(X::AbstractMatrix{T}) where {T}
    @assert has_graph(g.fg) "A GCNConv created without a graph must be given a FeaturedGraph as an input."
    W, b, σ = g.weight, g.bias, g.σ
    L = normalized_laplacian(g.fg, float(T); selfloop=true)
    L = convert(typeof(X), L)
    σ.(W * X * L .+ b)
end

function (g::GCNConv)(fg::FeaturedGraph)
    X = node_feature(fg)
    A = adjacency_matrix(fg)
    g.fg isa NullGraph || (g.fg.graph[] = A)
    L = normalized_laplacian(A, eltype(X); selfloop=true)
    X_ = g.σ.(g.weight * X * L .+ g.bias)
    FeaturedGraph(A, X_)
end

function Base.show(io::IO, l::GCNConv)
    in_channel = size(l.weight, ndims(l.weight))
    out_channel = size(l.weight, ndims(l.weight)-1)
    print(io, "GCNConv(G(V=", nv(l.graph))
    print(io, ", E), ", in_channel, "=>", out_channel)
    print(io, "GCNConv(", in_channel, "=>", out_channel)
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
                  init = glorot_uniform, T::DataType=Float32, bias::Bool=true, cache::Bool=true)
    b = bias ? init(ch[2]) : zeros(T, ch[2])
    fg = cache ? FeaturedGraph(adj) : NullGraph()
    ChebConv(init(ch[2], ch[1], k), b, fg, k, ch[1], ch[2])
end

function ChebConv(ch::Pair{<:Integer,<:Integer}, k::Integer;
                  init = glorot_uniform, T::DataType=Float32, bias::Bool=true, cache::Bool=true)
    b = bias ? init(ch[2]) : zeros(T, ch[2])
    fg = cache ? FeaturedGraph() : NullGraph()
    ChebConv(init(ch[2], ch[1], k), b, fg, k, ch[1], ch[2])
end

@functor ChebConv

function (c::ChebConv)(L̃::AbstractMatrix{S}, X::AbstractMatrix{T}) where {S<:Real, T<:Real}
    fin = c.in_channel
    @assert size(X, 1) == fin "Input feature size must match input channel size."
    N = size(L̃, 1)
    @assert size(X, 2) == N "Input vertex number must match Laplacian matrix size."
    fout = c.out_channel

    Z = similar(X, fin, N, c.k)
    Z[:,:,1] = X
    Z[:,:,2] = X * L̃
    for k = 3:c.k
        Z[:,:,k] = 2*view(Z, :, :, k-1)*L̃ - view(Z, :, :, k-2)
    end

    Y = view(c.weight, :, :, 1) * view(Z, :, :, 1)
    for k = 2:c.k
        Y += view(c.weight, :, :, k) * view(Z, :, :, k)
    end
    Y .+= c.bias
    return Y
end

function (c::ChebConv)(X::AbstractMatrix{T}) where {T<:Real}
    @assert has_graph(c.fg) "A ChebConv created without a graph must be given a FeaturedGraph as an input."
    g = graph(c.fg)
    L̃ = scaled_laplacian(g, T)
    c(L̃, X)
end

function (c::ChebConv)(fg::FeaturedGraph)
    @assert has_graph(fg) "A given FeaturedGraph must contain a graph."
    g = graph(fg)
    c.fg isa NullGraph || (c.fg.graph[] = g)
    L̃ = scaled_laplacian(adjacency_matrix(fg))
    X_ = c(L̃, node_feature(fg))
    FeaturedGraph(g, X_)
end

function Base.show(io::IO, l::ChebConv)
    print(io, "ChebConv(G(V=", size(l.L̃, 1))
    print(io, ", E), ", l.in_channel, "=>", l.out_channel)
    print(io, ", k=", l.k)
    print(io, ")")
end



"""
    GraphConv([graph, ]in=>out)
    GraphConv([graph, ]in=>out, aggr)

Graph neural network layer.

# Arguments
- `graph`: should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`,
`SimpleWeightedDiGraph` (from SimpleWeightedGraphs). Is optionnal so you can give a `FeaturedGraph` to
the layer instead of only the features.
- `in`: the dimension of input features.
- `out`: the dimension of output features.
- `bias::Bool=true`: keyword argument, whether to learn the additive bias.
- `aggr::Symbol=:add`: an aggregate function applied to the result of message function. `:add`, `:max` and `:mean` are available.
"""
struct GraphConv{V<:AbstractFeaturedGraph,T} <: MessagePassing
    fg::V
    weight1::AbstractMatrix{T}
    weight2::AbstractMatrix{T}
    bias::AbstractVector{T}
    aggr::Symbol
end

function GraphConv(el::AbstractVector{<:AbstractVector{<:Integer}},
                   ch::Pair{<:Integer,<:Integer}, aggr=:add;
                   init = glorot_uniform, bias::Bool=true)
    b = bias ? init(ch[2]) : zeros(T, ch[2])
    fg = FeaturedGraph(el)
    GraphConv(fg, init(ch[2], ch[1]), init(ch[2], ch[1]), b, aggr)
end

function GraphConv(adj::AbstractMatrix, ch::Pair{<:Integer,<:Integer}, aggr=:add;
                   init = glorot_uniform, bias::Bool=true, T::DataType=Float32)
    b = bias ? init(ch[2]) : zeros(T, ch[2])
    fg = FeaturedGraph(neighbors(adj))
    GraphConv(fg, init(ch[2], ch[1]), init(ch[2], ch[1]), b, aggr)
end

function GraphConv(ch::Pair{<:Integer,<:Integer}, aggr=:add;
                   init = glorot_uniform, bias::Bool=true, T::DataType=Float32)
    b = bias ? init(ch[2]) : zeros(T, ch[2])
    GraphConv(NullGraph(), init(ch[2], ch[1]), init(ch[2], ch[1]), b, aggr)
end

@functor GraphConv

message(g::GraphConv, x_i, x_j::AbstractVector, e_ij) = g.weight2 * x_j
update(g::GraphConv, m::AbstractVector, x::AbstractVector) = g.weight1*x .+ m .+ g.bias
function (g::GraphConv)(X::AbstractMatrix)
    @assert has_graph(g.fg) "A GraphConv created without a graph must be given a FeaturedGraph as an input."
    fg = FeaturedGraph(graph(g.fg), X)
    fg_ = g(fg)
    node_feature(fg_)
end
(g::GraphConv)(fg::FeaturedGraph) = propagate(g, fg, :add)

function Base.show(io::IO, l::GraphConv)
    in_channel = size(l.weight1, ndims(l.weight1))
    out_channel = size(l.weight1, ndims(l.weight1)-1)
    print(io, "GraphConv(G(V=", nv(l.fg), ", E=", ne(l.fg))
    print(io, "), ", in_channel, "=>", out_channel)
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
struct GATConv{V<:AbstractFeaturedGraph, T <: Real} <: MessagePassing
    fg::V
    weight::AbstractMatrix{T}
    bias::AbstractVector{T}
    a::AbstractArray{T,3}
    negative_slope::Real
    channel::Pair{<:Integer,<:Integer}
    heads::Integer
    concat::Bool
end

function GATConv(adj::AbstractMatrix, ch::Pair{<:Integer,<:Integer}; heads::Integer=1,
                 concat::Bool=true, negative_slope::Real=0.2, init=glorot_uniform,
                 bias::Bool=true, T::DataType=Float32)
    N = size(adj, 1)
    w = init(ch[2]*heads, ch[1])
    b = bias ? init(ch[2]*heads) : zeros(T, ch[2]*heads)
    a = init(2*ch[2], heads, 1)
    fg = FeaturedGraph(neighbors(adj))
    GATConv(fg, w, b, a, negative_slope, ch, heads, concat)
end

function GATConv(ch::Pair{<:Integer,<:Integer}; heads::Integer=1,
                 concat::Bool=true, negative_slope::Real=0.2, init=glorot_uniform,
                 bias::Bool=true, T::DataType=Float32)
    w = init(ch[2]*heads, ch[1])
    b = bias ? init(ch[2]*heads) : zeros(T, ch[2]*heads)
    a = init(2*ch[2], heads, 1)
    GATConv(NullGraph(), w, b, a, negative_slope, ch, heads, concat)
end

@functor GATConv

function message(g::GATConv, x_i::AbstractArray, x_j::AbstractArray, e_ij)
    x_i = reshape(x_i, g.channel[2], g.heads, :)
    x_j = reshape(x_j, g.channel[2], g.heads, :)
    n = size(x_j, 3)
    α = cat(repeat(x_i, outer=(1,1,n)), x_j+zero(x_j), dims=1) .* g.a
    α = reshape(sum(α, dims=1), g.heads, n)
    α = leakyrelu.(α, g.negative_slope)
    α = _softmax(α)
    x_j .*= reshape(α, 1, g.heads, n)
    reshape(x_j, g.channel[2]*g.heads, :)
end

function update(g::GATConv, M::AbstractArray, x::AbstractVector)
    if !g.concat
        M = mean(M, dims=2)
    end
    return M .+ g.bias
end

function (g::GATConv)(X::AbstractMatrix)
    @assert has_graph(g.fg) "A GATConv created without a graph must be given a FeaturedGraph as an input."
    fg = FeaturedGraph(graph(g.fg), g.weight*X)
    fg_ = g(fg)
    node_feature(fg_)
end
(g::GATConv)(fg::FeaturedGraph) = propagate(g, fg, :add)


function _softmax(xs)
    xs = exp.(xs)
    s = sum(xs, dims=2)
    return xs ./ s
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
                        aggr=:add, init=glorot_uniform)
    N = size(adj, 1)
    w = init(out_ch, out_ch, num_layers)
    gru = GRUCell(out_ch, out_ch)
    fg = FeaturedGraph(neighbors(adj))
    GatedGraphConv(fg, w, gru, out_ch, num_layers, aggr)
end

function GatedGraphConv(out_ch::Integer, num_layers::Integer;
                        aggr=:add, init=glorot_uniform)
    w = init(out_ch, out_ch, num_layers)
    gru = GRUCell(out_ch, out_ch)
    GatedGraphConv(NullGraph(), w, gru, out_ch, num_layers, aggr)
end

@functor GatedGraphConv

message(g::GatedGraphConv, x_i, x_j::AbstractVector, e_ij) = x_j
update(g::GatedGraphConv, m::AbstractVector, x) = m

function (g::GatedGraphConv)(X::AbstractMatrix)
    @assert has_graph(g.fg) "A GraphConv created without a graph must be given a FeaturedGraph as an input."
    fg = FeaturedGraph(graph(g.fg), X)
    fg_ = g(fg)
    node_feature(fg_)
end

function (g::GatedGraphConv{V,T})(fg::FeaturedGraph) where {V,T<:Real}
    H = node_feature(fg)
    m, n = size(H)
    @assert (m <= g.out_ch) "number of input features must less or equals to output features."
    (m < g.out_ch) && (H = vcat(H, zeros(T, g.out_ch - m, n)))

    for i = 1:g.num_layers
        M = view(g.weight, :, :, i) * H
        M = propagate(g, fg, g.aggr)
        H, _ = g.gru(H, M)
    end
    FeaturedGraph(graph(fg), H)
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
    fg = FeaturedGraph(neighbors(adj))
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
    fg = FeaturedGraph(graph(e.fg), X)
    fg_ = e(fg)
    node_feature(fg_)
end

(e::EdgeConv)(fg::FeaturedGraph) = propagate(e, fg, e.aggr)

function Base.show(io::IO, l::EdgeConv)
    print(io, "EdgeConv(G(V=", nv(l.fg), ", E=", ne(l.fg))
    print(io, "), ", l.nn)
    print(io, ", aggr=", AGGR2STR[l.aggr])
    print(io, ")")
end
