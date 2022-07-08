"""
    GCNConv(in => out, σ=identity; bias=true, init=glorot_uniform)

Graph convolutional layer. The input to the layer is a node feature array `X`
of size `(num_features, num_nodes)`.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.

# Examples

```jldoctest
julia> gc = GCNConv(1024=>256, relu)
GCNConv(1024 => 256, relu)
```

See also [`WithGraph`](@ref) for training layer with static graph.
"""
struct GCNConv{A<:AbstractMatrix,B,F} <: AbstractGraphLayer
    weight::A
    bias::B
    σ::F
end

function GCNConv(ch::Pair{Int,Int}, σ=identity;
                 init=glorot_uniform, bias::Bool=true)
    in, out = ch
    W = init(out, in)
    b = Flux.create_bias(W, bias, out)
    return GCNConv(W, b, σ)
end

@functor GCNConv

function (l::GCNConv)(Ã::AbstractMatrix, X::AbstractArray)
    z = _matmul(l.weight, _matmul(X, Ã))
    return l.σ.(z .+ l.bias)
end

# For variable graph
function (l::GCNConv)(fg::AbstractFeaturedGraph)
    nf = node_feature(fg)
    Ã = Zygote.ignore() do
        GraphSignals.normalized_adjacency_matrix(fg, eltype(nf); selfloop=true)
    end
    return ConcreteFeaturedGraph(fg, nf = l(Ã, nf))
end

# For static graph
WithGraph(fg::AbstractFeaturedGraph, l::GCNConv) =
    WithGraph(GraphSignals.normalized_adjacency_matrix(fg, eltype(l.weight); selfloop=true), l)

function (wg::WithGraph{<:GCNConv})(X::AbstractArray)
    Ã = wg.graph
    return wg.layer(Ã, X)
end

function Base.show(io::IO, l::GCNConv)
    out, in = size(l.weight)
    print(io, "GCNConv($in => $out")
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end


"""
    ChebConv(in=>out, k; bias=true, init=glorot_uniform)

Chebyshev spectral graph convolutional layer.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `k`: The order of Chebyshev polynomial.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.

# Examples

```jldoctest
julia> cc = ChebConv(1024=>256, 5, relu)
ChebConv(1024 => 256, k=5, relu)
```

See also [`WithGraph`](@ref) for training layer with static graph.
"""
struct ChebConv{A<:AbstractArray{<:Number,3},B,F} <: AbstractGraphLayer
    weight::A
    bias::B
    k::Int
    σ::F
end

function ChebConv(ch::Pair{Int,Int}, k::Int, σ=identity;
                  init=glorot_uniform, bias::Bool=true)
    in, out = ch
    W = init(out, in, k)
    b = Flux.create_bias(W, bias, out)
    ChebConv(W, b, k, σ)
end

@functor ChebConv

Flux.trainable(l::ChebConv) = (l.weight, l.bias)

function (l::ChebConv)(L̃::AbstractMatrix, X::AbstractArray)
    Z_prev = X
    Z = _matmul(X, L̃)
    Y = _matmul(view(l.weight,:,:,1), Z_prev)
    Y += _matmul(view(l.weight,:,:,2), Z)
    for k = 3:l.k
        Z, Z_prev = 2 .* _matmul(Z, L̃) .- Z_prev, Z
        Y += _matmul(view(l.weight,:,:,k), Z)
    end
    return l.σ.(Y .+ l.bias)
end

# For variable graph
function (l::ChebConv)(fg::AbstractFeaturedGraph)
    nf = node_feature(fg)
    GraphSignals.check_num_nodes(fg, nf)
    @assert size(nf, 1) == size(l.weight, 2) "Input feature size must match input channel size."
    
    L̃ = Zygote.ignore() do
        GraphSignals.scaled_laplacian(fg, eltype(nf))
    end
    return ConcreteFeaturedGraph(fg, nf = l(L̃, nf))
end

# For static graph
WithGraph(fg::AbstractFeaturedGraph, l::ChebConv) =
    WithGraph(GraphSignals.scaled_laplacian(fg, eltype(l.weight)), l)

function (wg::WithGraph{<:ChebConv})(X::AbstractArray)
    L̃ = wg.graph
    return wg.layer(L̃, X)
end

function Base.show(io::IO, l::ChebConv)
    out, in, k = size(l.weight)
    print(io, "ChebConv(", in, " => ", out)
    print(io, ", k=", k)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end


"""
    GraphConv(in => out, σ=identity, aggr=+; bias=true, init=glorot_uniform)

Graph neural network layer.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `aggr`: An aggregate function applied to the result of message function. `+`, `-`,
`*`, `/`, `max`, `min` and `mean` are available.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.

# Examples

```jldoctest
julia> GraphConv(1024=>256, relu)
GraphConv(1024 => 256, relu, aggr=+)

julia> GraphConv(1024=>256, relu, *)
GraphConv(1024 => 256, relu, aggr=*)
```

See also [`WithGraph`](@ref) for training layer with static graph.
"""
struct GraphConv{A<:AbstractMatrix,B,F,O} <: MessagePassing
    weight1::A
    weight2::A
    bias::B
    σ::F
    aggr::O
end

function GraphConv(ch::Pair{Int,Int}, σ=identity, aggr=+;
                   init=glorot_uniform, bias::Bool=true)
    in, out = ch
    W1 = init(out, in)
    W2 = init(out, in)
    b = Flux.create_bias(W1, bias, out)
    GraphConv(W1, W2, b, σ, aggr)
end

@functor GraphConv

Flux.trainable(l::GraphConv) = (l.weight1, l.weight2, l.bias)

message(gc::GraphConv, x_i, x_j::AbstractArray, e_ij) = _matmul(gc.weight2, x_j)

update(gc::GraphConv, m::AbstractArray, x::AbstractArray) = gc.σ.(_matmul(gc.weight1, x) .+ m .+ gc.bias)

# For variable graph
function (l::GraphConv)(fg::AbstractFeaturedGraph)
    nf = node_feature(fg)
    GraphSignals.check_num_nodes(fg, nf)
    _, V, _ = propagate(l, graph(fg), nothing, nf, nothing, l.aggr, nothing, nothing)
    return ConcreteFeaturedGraph(fg, nf=V)
end

# For static graph
function (gc::GraphConv)(el::NamedTuple, x::AbstractArray)
    GraphSignals.check_num_nodes(el.N, x)
    _, x, _ = propagate(gc, el, nothing, x, nothing, gc.aggr, nothing, nothing)
    return x
end

function Base.show(io::IO, l::GraphConv)
    in_channel = size(l.weight1, ndims(l.weight1))
    out_channel = size(l.weight1, ndims(l.weight1)-1)
    print(io, "GraphConv(", in_channel, " => ", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end



"""
    GATConv(in => out, σ=identity; heads=1, concat=true,
            init=glorot_uniform, bias=true, negative_slope=0.2)

Graph attentional layer.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `bias::Bool`: Keyword argument, whether to learn the additive bias.
- `σ`: Activation function.
- `heads`: Number attention heads 
- `concat`: Concatenate layer output or not. If not, layer output is averaged.
- `negative_slope::Real`: Keyword argument, the parameter of LeakyReLU.

# Examples

```jldoctest
julia> GATConv(1024=>256, relu)
GATConv(1024=>256, relu, heads=1, concat=true, LeakyReLU(λ=0.2))

julia> GATConv(1024=>256, relu, heads=4)
GATConv(1024=>1024, relu, heads=4, concat=true, LeakyReLU(λ=0.2))

julia> GATConv(1024=>256, relu, heads=4, concat=false)
GATConv(1024=>1024, relu, heads=4, concat=false, LeakyReLU(λ=0.2))

julia> GATConv(1024=>256, relu, negative_slope=0.1f0)
GATConv(1024=>256, relu, heads=1, concat=true, LeakyReLU(λ=0.1))
```

See also [`WithGraph`](@ref) for training layer with static graph.
"""
struct GATConv{T,A<:AbstractMatrix{T},B,F} <: MessagePassing
    weight::A
    bias::B
    a::A
    σ::F
    negative_slope::T
    channel::Pair{Int, Int}
    heads::Int
    concat::Bool
end

function GATConv(ch::Pair{Int,Int}, σ=identity; heads::Int=1, concat::Bool=true,
                 negative_slope=0.2f0, init=glorot_uniform, bias::Bool=true)
    in, out = ch             
    W = init(out*heads, in)
    b = Flux.create_bias(W, bias, out*heads)
    a = init(2*out, heads)
    GATConv(W, b, a, σ, negative_slope, ch, heads, concat)
end

@functor GATConv

Flux.trainable(l::GATConv) = (l.weight, l.bias, l.a)

# neighbor attention
function update_batch_edge(gat::GATConv, el::NamedTuple, E, X::AbstractMatrix, u)
    X = reshape(X, size(X)..., 1)
    M = update_batch_edge(gat, el, E, X, u)
    return reshape(M, size(M)[1:2]...)
end

function update_batch_edge(gat::GATConv, el::NamedTuple, E, X::AbstractArray, u)
    Xi, Xj = _gather(X, el.xs), _gather(X, el.nbrs)
    _, nb, bch_sz = size(Xj)
    heads = gat.heads
    Q = reshape(NNlib.batched_mul(gat.weight, Xi), :, heads, nb, bch_sz)  # dims: (out, heads, nb, bch_sz)
    K = reshape(NNlib.batched_mul(gat.weight, Xj), :, heads, nb, bch_sz)
    V = reshape(NNlib.batched_mul(gat.weight, Xj), :, heads, nb, bch_sz)
    QK = vcat(Q, K)  # dims: (2out, heads, nb, bch_sz)
    A = leakyrelu.(sum(QK .* gat.a, dims=1), gat.negative_slope)  # dims: (1, heads, nb, bch_sz)
    α = indexed_softmax(A, el.xs, el.N, dims=3)  # dims: (1, heads, nb, bch_sz)
    N = incidence_matrix(el.xs, el.N)
    Y = NNlib.batched_mul(reshape(V .* α, :, nb, bch_sz), N)  # dims: (out*heads, N, bch_sz)
    return Y
end

# graph attention
aggregate_neighbors(::GATConv, el::NamedTuple, aggr, E::AbstractArray) = E  # dims: (out*heads, N, [bch_sz])
aggregate_neighbors(::GATConv, el::NamedTuple, aggr, E::AbstractMatrix) = E

function update(gat::GATConv, M::AbstractArray, X)
    M = M .+ gat.bias
    if gat.concat || gat.heads == 1
        M = gat.σ.(M)  # dims: (out*heads, N, [bch_sz])
    else
        dims = size(M)[2:end]
        M = reshape(M, :, gat.heads, dims...)
        M = gat.σ.(mean(M, dims=2))
        M = reshape(M, :, dims...)  # dims: (out, N, [bch_sz])
    end
    return M
end

# For variable graph
function (l::GATConv)(fg::AbstractFeaturedGraph)
    X = node_feature(fg)
    GraphSignals.check_num_nodes(fg, X)
    sg = graph(fg)
    @assert Zygote.ignore(() -> GraphSignals.has_all_self_loops(sg)) "a vertex must have self loop (receive a message from itself)."
    el = GraphSignals.to_namedtuple(sg)
    _, V, _ = propagate(l, el, nothing, X, nothing, hcat, nothing, nothing)
    return ConcreteFeaturedGraph(fg, nf=V)
end

# For static graph
function (l::GATConv)(el::NamedTuple, X::AbstractArray)
    GraphSignals.check_num_nodes(el.N, X)
    # TODO: should have self loops check for el
    _, V, _ = propagate(l, el, nothing, X, nothing, hcat, nothing, nothing)
    return V
end

function Base.show(io::IO, l::GATConv)
    in_channel = size(l.weight, ndims(l.weight))
    out_channel = size(l.weight, ndims(l.weight)-1)
    print(io, "GATConv(", in_channel, "=>", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", heads=", l.heads)
    print(io, ", concat=", l.concat)
    print(io, ", LeakyReLU(λ=", l.negative_slope)
    print(io, "))")
end


"""
    GATv2Conv(in => out, σ=identity; heads=1, concat=true,
              init=glorot_uniform, negative_slope=0.2)

Graph attentional layer v2.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `heads`: Number attention heads 
- `concat`: Concatenate layer output or not. If not, layer output is averaged.
- `negative_slope::Real`: Keyword argument, the parameter of LeakyReLU.

# Examples

```jldoctest
julia> GATv2Conv(1024=>256, relu)
GATv2Conv(1024=>256, relu, heads=1, concat=true, LeakyReLU(λ=0.2))

julia> GATv2Conv(1024=>256, relu, heads=4)
GATv2Conv(1024=>1024, relu, heads=4, concat=true, LeakyReLU(λ=0.2))

julia> GATv2Conv(1024=>256, relu, heads=4, concat=false)
GATv2Conv(1024=>1024, relu, heads=4, concat=false, LeakyReLU(λ=0.2))

julia> GATv2Conv(1024=>256, relu, negative_slope=0.1f0)
GATv2Conv(1024=>256, relu, heads=1, concat=true, LeakyReLU(λ=0.1))
```

See also [`WithGraph`](@ref) for training layer with static graph.
"""
struct GATv2Conv{T, A<:AbstractMatrix{T}, B, F} <: MessagePassing
    wi::A
    wj::A
    biasi::B
    biasj::B
    a::A
    σ::F
    negative_slope::T
    channel::Pair{Int, Int}
    heads::Int
    concat::Bool
end

function GATv2Conv(ch::Pair{Int,Int}, σ=identity; heads::Int=1, concat::Bool=true,
                   negative_slope=0.2f0, bias::Bool=true, init=glorot_uniform)
    in, out = ch
    wi = init(out*heads, in)
    wj = init(out*heads, in)
    bi = Flux.create_bias(wi, bias, out*heads)
    bj = Flux.create_bias(wj, bias, out*heads)
    a = init(out, heads)
    GATv2Conv(wi, wj, bi, bj, a, σ, negative_slope, ch, heads, concat)
end

@functor GATv2Conv

Flux.trainable(l::GATv2Conv) = (l.wi, l.wj, l.biasi, l.biasj, l.a)

function update_batch_edge(gat::GATv2Conv, el::NamedTuple, E, X::AbstractMatrix, u)
    X = reshape(X, size(X)..., 1)
    M = update_batch_edge(gat, el, E, X, u)
    return reshape(M, size(M)[1:2]...)
end

function update_batch_edge(gat::GATv2Conv, el::NamedTuple, E, X::AbstractArray, u)
    Xi, Xj = _gather(X, el.xs), _gather(X, el.nbrs)
    _, nb, bch_sz = size(Xj)
    heads = gat.heads
    Q = reshape(NNlib.batched_mul(gat.wi, Xi) .+ gat.biasi, :, heads, nb, bch_sz)  # dims: (out, heads, nb, bch_sz)
    K = reshape(NNlib.batched_mul(gat.wj, Xj) .+ gat.biasj, :, heads, nb, bch_sz)
    V = reshape(NNlib.batched_mul(gat.wj, Xj) .+ gat.biasj, :, heads, nb, bch_sz)
    QK = Q + K  # dims: (out, heads, nb, bch_sz)
    A = leakyrelu.(sum(QK .* gat.a, dims=1), gat.negative_slope)  # dims: (1, heads, nb, bch_sz)
    α = indexed_softmax(A, el.xs, el.N, dims=3)  # dims: (1, heads, nb, bch_sz)
    N = incidence_matrix(el.xs, el.N)
    Y = NNlib.batched_mul(reshape(V .* α, :, nb, bch_sz), N)  # dims: (out*heads, N, bch_sz)
    return Y
end

aggregate_neighbors(::GATv2Conv, el::NamedTuple, aggr, E::AbstractArray) = E  # dims: (out*heads, N, [bch_sz])
aggregate_neighbors(::GATv2Conv, el::NamedTuple, aggr, E::AbstractMatrix) = E

function update(gat::GATv2Conv, M::AbstractArray, X)
    if gat.concat || gat.heads == 1
        M = gat.σ.(M)  # dims: (out*heads, N, [bch_sz])
    else
        dims = size(M)[2:end]
        M = reshape(M, :, gat.heads, dims...)
        M = gat.σ.(mean(M, dims=2))
        M = reshape(M, :, dims...)  # dims: (out, N, [bch_sz])
    end
    return M
end

# For variable graph
function (l::GATv2Conv)(fg::AbstractFeaturedGraph)
    X = node_feature(fg)
    GraphSignals.check_num_nodes(fg, X)
    sg = graph(fg)
    @assert Zygote.ignore(() -> GraphSignals.has_all_self_loops(sg)) "a vertex must have self loop (receive a message from itself)."
    el = GraphSignals.to_namedtuple(sg)
    _, V, _ = propagate(l, el, nothing, X, nothing, hcat, nothing, nothing)
    return ConcreteFeaturedGraph(fg, nf=V)
end

# For static graph
function (l::GATv2Conv)(el::NamedTuple, X::AbstractArray)
    GraphSignals.check_num_nodes(el.N, X)
    # TODO: should have self loops check for el
    _, V, _ = propagate(l, el, nothing, X, nothing, hcat, nothing, nothing)
    return V
end

function Base.show(io::IO, l::GATv2Conv)
    in_channel = size(l.wi, ndims(l.wi))
    out_channel = size(l.wi, ndims(l.wi)-1)
    print(io, "GATv2Conv(", in_channel, "=>", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", heads=", l.heads)
    print(io, ", concat=", l.concat)
    print(io, ", LeakyReLU(λ=", l.negative_slope)
    print(io, "))")
end



"""
    GatedGraphConv([fg,] out, num_layers; aggr=+, init=glorot_uniform)

Gated graph convolution layer.

# Arguments

- `out`: The dimension of output features.
- `num_layers`: The number of gated recurrent unit.
- `aggr`: An aggregate function applied to the result of message function. `+`, `-`,
`*`, `/`, `max`, `min` and `mean` are available.

# Examples

```jldoctest
julia> GatedGraphConv(256, 4)
GatedGraphConv((256 => 256)^4, aggr=+)

julia> GatedGraphConv(256, 4, aggr=*)
GatedGraphConv((256 => 256)^4, aggr=*)
```

See also [`WithGraph`](@ref) for training layer with static graph.
"""
struct GatedGraphConv{A<:AbstractArray{<:Number,3},R,O} <: MessagePassing
    weight::A
    gru::R
    out_ch::Int
    num_layers::Int
    aggr::O
end

function GatedGraphConv(out_ch::Int, num_layers::Int; aggr=+, init=glorot_uniform)
    w = init(out_ch, out_ch, num_layers)
    gru = GRUCell(out_ch, out_ch)
    GatedGraphConv(w, gru, out_ch, num_layers, aggr)
end

@functor GatedGraphConv

Flux.trainable(l::GatedGraphConv) = (l.weight, l.gru)

message(ggc::GatedGraphConv, x_i, x_j::AbstractArray, e_ij) = x_j

update(ggc::GatedGraphConv, m::AbstractArray, x) = m

# For variable graph
function (l::GatedGraphConv)(fg::AbstractFeaturedGraph)
    nf = node_feature(fg)
    GraphSignals.check_num_nodes(fg, nf)
    V = l(GraphSignals.to_namedtuple(fg), nf)
    return ConcreteFeaturedGraph(fg, nf=V)
end

# For static graph
function (l::GatedGraphConv)(el::NamedTuple, H::AbstractArray{T}) where {T<:Real}
    GraphSignals.check_num_nodes(el.N, H)
    m, n = size(H)[1:2]
    @assert (m <= l.out_ch) "number of input features must less or equals to output features."
    if m < l.out_ch
        Hpad = Zygote.ignore() do
            fill!(similar(H, T, l.out_ch - m, n, size(H)[3:end]...), 0)
        end
        H = vcat(H, Hpad)
    end
    for i = 1:l.num_layers
        M = _matmul(view(l.weight, :, :, i), H)
        _, M = propagate(l, el, nothing, M, nothing, l.aggr, nothing, nothing)
        H, _ = l.gru(H, M)
    end
    return H
end

function Base.show(io::IO, l::GatedGraphConv)
    print(io, "GatedGraphConv(($(l.out_ch) => $(l.out_ch))^$(l.num_layers)")
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end



"""
    EdgeConv(nn; aggr=max)

Edge convolutional layer.

# Arguments

- `nn`: A neural network (e.g. a Dense layer or a MLP).
- `aggr`: An aggregate function applied to the result of message function.
`+`, `max` and `mean` are available.

# Examples

```jldoctest
julia> EdgeConv(Dense(1024, 256, relu))
EdgeConv(Dense(1024 => 256, relu), aggr=max)

julia> EdgeConv(Dense(1024, 256, relu), aggr=+)
EdgeConv(Dense(1024 => 256, relu), aggr=+)
```

See also [`WithGraph`](@ref) for training layer with static graph.
"""
struct EdgeConv{N,O} <: MessagePassing
    nn::N
    aggr::O
end

EdgeConv(nn; aggr=max) = EdgeConv(nn, aggr)

@functor EdgeConv

Flux.trainable(l::EdgeConv) = (l.nn,)

message(ec::EdgeConv, x_i::AbstractArray, x_j::AbstractArray, e_ij) = ec.nn(vcat(x_i, x_j .- x_i))
update(ec::EdgeConv, m::AbstractArray, x) = m

# For variable graph
function (l::EdgeConv)(fg::AbstractFeaturedGraph)
    nf = node_feature(fg)
    GraphSignals.check_num_nodes(fg, nf)
    _, V, _ = propagate(l, graph(fg), nothing, nf, nothing, l.aggr, nothing, nothing)
    return ConcreteFeaturedGraph(fg, nf=V)
end

# For static graph
function (l::EdgeConv)(el::NamedTuple, X::AbstractArray)
    GraphSignals.check_num_nodes(el.N, X)
    _, X, _ = propagate(l, el, nothing, X, nothing, l.aggr, nothing, nothing)
    return X
end

function Base.show(io::IO, l::EdgeConv)
    print(io, "EdgeConv(", l.nn, ", aggr=", l.aggr, ")")
end


"""
    GINConv(nn, [eps=0])

    Graph Isomorphism Network.

# Arguments

- `nn`: A neural network/layer.
- `eps`: Weighting factor.

# Examples

```jldoctest
julia> GINConv(Dense(1024, 256, relu))
GINConv(Dense(1024 => 256, relu), ϵ=0.0)

julia> GINConv(Dense(1024, 256, relu), 1.f-6)
GINConv(Dense(1024 => 256, relu), ϵ=1.0e-6)
```

See also [`WithGraph`](@ref) for training layer with static graph.
"""
struct GINConv{N,R<:Real} <: MessagePassing
    nn::N
    eps::R
end

GINConv(nn, eps=0f0) = GINConv(nn, eps)

@functor GINConv

Flux.trainable(g::GINConv) = (g.nn,)

message(g::GINConv, x_i::AbstractArray, x_j::AbstractArray) = x_j 
update(g::GINConv, m::AbstractArray, x::AbstractArray) = g.nn((1 + g.eps) * x + m)

# For variable graph
function (l::GINConv)(fg::AbstractFeaturedGraph)
    nf = node_feature(fg)
    GraphSignals.check_num_nodes(fg, nf)
    _, V, _ = propagate(l, graph(fg), nothing, nf, nothing, +, nothing, nothing)
    return ConcreteFeaturedGraph(fg, nf=V)
end

# For static graph
function (l::GINConv)(el::NamedTuple, x::AbstractArray)
    GraphSignals.check_num_nodes(el.N, x)
    _, V, _ = propagate(l, el, nothing, x, nothing, +, nothing, nothing)
    return V
end

function Base.show(io::IO, l::GINConv)
    print(io, "GINConv(", l.nn, ", ϵ=", l.eps, ")")
end


"""
    CGConv((node_dim, edge_dim), init, bias=true)

Crystal Graph Convolutional network. Uses both node and edge features.

# Arguments

- `node_dim`: Dimensionality of the input node features. Also is necessarily the output dimensionality.
- `edge_dim`: Dimensionality of the input edge features.
- `init`: Initialization algorithm for each of the weight matrices
- `bias`: Whether or not to learn an additive bias parameter.

# Examples

```jldoctest
julia> CGConv((128, 32))
CGConv(node dim=128, edge dim=32)
```

See also [`WithGraph`](@ref) for training layer with static graph.
"""
struct CGConv{A<:AbstractMatrix,B} <: MessagePassing
    Wf::A
    Ws::A
    bf::B
    bs::B
end

@functor CGConv

Flux.trainable(l::CGConv) = (l.Wf, l.Ws, l.bf, l.bs)

function CGConv(dims::NTuple{2,Int}; init=glorot_uniform, bias=true)
    node_dim, edge_dim = dims
    Wf = init(node_dim, 2*node_dim + edge_dim)
    Ws = init(node_dim, 2*node_dim + edge_dim)
    bf = Flux.create_bias(Wf, bias, node_dim)
    bs = Flux.create_bias(Ws, bias, node_dim)
    return CGConv(Wf, Ws, bf, bs)
end

function message(c::CGConv, x_i::AbstractArray, x_j::AbstractArray, e::AbstractArray)
    z = vcat(x_i, x_j, e)

    return σ.(_matmul(c.Wf, z) .+ c.bf) .* softplus.(_matmul(c.Ws, z) .+ c.bs)
end

update(c::CGConv, m::AbstractArray, x) = x + m

# For variable graph
function (l::CGConv)(fg::AbstractFeaturedGraph)
    X = node_feature(fg)
    E = edge_feature(fg)
    GraphSignals.check_num_nodes(fg, X)
    GraphSignals.check_num_edges(fg, E)
    _, V, _ = propagate(l, graph(fg), E, X, nothing, +, nothing, nothing)
    return ConcreteFeaturedGraph(fg, nf=V)
end

# For static graph
function (l::CGConv)(el::NamedTuple, X::AbstractArray, E::AbstractArray)
    GraphSignals.check_num_nodes(el.N, X)
    GraphSignals.check_num_edges(el.E, E)
    _, V, _ = propagate(l, el, E, X, nothing, +, nothing, nothing)
    return V
end

function Base.show(io::IO, l::CGConv)
    node_dim, d = size(l.Wf)
    edge_dim = d - 2*node_dim
    print(io, "CGConv(node dim=", node_dim, ", edge dim=", edge_dim, ")")
end

"""
    SAGEConv(in => out, σ=identity, aggr=mean; normalize=true, project=false,
             bias=true, num_sample=10, init=glorot_uniform)

SAmple and aggreGatE convolutional layer for GraphSAGE network.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `aggr`: An aggregate function applied to the result of message function. `mean`, `max`,
`LSTM` and `GCNConv` are available.
- `normalize::Bool`: Whether to normalize features across all nodes or not.
- `project::Bool`: Whether to project, i.e. `Dense(in, in)`, before aggregation.
- `bias`: Add learnable bias.
- `num_sample::Int`: Number of samples for each node from their neighbors.
- `init`: Weights' initializer.

# Examples

```jldoctest
julia> SAGEConv(1024=>256, relu)
SAGEConv(1024 => 256, relu, aggr=mean, normalize=true, #sample=10)

julia> SAGEConv(1024=>256, relu, num_sample=5)
SAGEConv(1024 => 256, relu, aggr=mean, normalize=true, #sample=5)

julia> MeanAggregator(1024=>256, relu, normalize=false)
SAGEConv(1024 => 256, relu, aggr=mean, normalize=false, #sample=10)

julia> MeanPoolAggregator(1024=>256, relu)
SAGEConv(1024 => 256, relu, project=Dense(1024 => 1024), aggr=mean, normalize=true, #sample=10)

julia> MaxPoolAggregator(1024=>256, relu)
SAGEConv(1024 => 256, relu, project=Dense(1024 => 1024), aggr=max, normalize=true, #sample=10)

julia> LSTMAggregator(1024=>256, relu)
SAGEConv(1024 => 256, relu, aggr=LSTMCell(1024 => 1024), normalize=true, #sample=10)
```

See also [`WithGraph`](@ref) for training layer with static graph and [`MeanAggregator`](@ref),
[`MeanPoolAggregator`](@ref), [`MaxPoolAggregator`](@ref) and [`LSTMAggregator`](@ref).
"""
struct SAGEConv{A,B,F,P,O} <: MessagePassing
    weight1::A
    weight2::A
    bias::B
    σ::F
    proj::P
    aggr::O
    normalize::Bool
    num_sample::Int
end

function SAGEConv(ch::Pair{Int,Int}, σ=identity, aggr=mean;
                  normalize::Bool=true, project::Bool=false, bias::Bool=true,
                  num_sample::Int=10, init=glorot_uniform)
    in, out = ch
    weight1 = init(out, in)
    weight2 = init(out, in)
    bias = Flux.create_bias(weight1, bias, out)
    proj = project ? Dense(in, in) : identity
    return SAGEConv(weight1, weight2, bias, σ, proj, aggr, normalize, num_sample)
end

@functor SAGEConv

message(l::SAGEConv, x_i, x_j::AbstractArray, e) = l.proj(x_j)

function aggregate_neighbors(l::SAGEConv, el::NamedTuple, aggr, E)
    batch_size = size(E)[end]
    sample_idx = sample_node_index(E, l.num_sample; dims=2)
    idx = ntuple(i -> (i == 2) ? sample_idx : Colon(), ndims(E))
    dstsize = (size(E, 1), el.N, batch_size)  # ensure outcome has the same dimension as x in update
    xs = batched_index(el.xs[sample_idx], batch_size)
    Ē = _scatter(aggr, E[idx...], xs, dstsize)
    return Ē
end

function aggregate_neighbors(l::SAGEConv, el::NamedTuple, aggr, E::AbstractMatrix)
    sample_idx = sample_node_index(E, l.num_sample; dims=2)
    idx = ntuple(i -> (i == 2) ? sample_idx : Colon(), ndims(E))
    dstsize = (size(E, 1), el.N)  # ensure outcome has the same dimension as x in update
    Ē = _scatter(aggr, E[idx...], el.xs[sample_idx], dstsize)
    return Ē
end

aggregate_neighbors(::SAGEConv, el::NamedTuple, lstm::Flux.LSTMCell, E::AbstractArray) =
    throw(ArgumentError("SAGEConv with LSTM aggregator does not support batch learning."))

function aggregate_neighbors(::SAGEConv, el::NamedTuple, lstm::Flux.LSTMCell, E::AbstractMatrix)
    sample_idx = sample_node_index(E, el.N; dims=2)
    idx = ntuple(i -> (i == 2) ? sample_idx : Colon(), ndims(E))
    state, Ē = lstm(lstm.state0, E[idx...])
    return Ē
end

function update(l::SAGEConv, m::AbstractArray, x::AbstractArray)
    y = l.σ.(_matmul(l.weight1, x) + _matmul(l.weight2, m) .+ l.bias)
    l.normalize && (y = l2normalize(y; dims=2))  # across all nodes
    return y
end

# For variable graph
function (l::SAGEConv)(fg::AbstractFeaturedGraph)
    nf = node_feature(fg)
    GraphSignals.check_num_nodes(fg, nf)
    _, V, _ = propagate(l, graph(fg), nothing, nf, nothing, l.aggr, nothing, nothing)
    return ConcreteFeaturedGraph(fg, nf=V)
end

# For static graph
function (l::SAGEConv)(el::NamedTuple, x::AbstractArray)
    GraphSignals.check_num_nodes(el.N, x)
    _, V, _ = propagate(l, el, nothing, x, nothing, l.aggr, nothing, nothing)
    return V
end

function Base.show(io::IO, l::SAGEConv)
    out_channel, in_channel = size(l.weight1)
    print(io, "SAGEConv(", in_channel, " => ", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    l.proj == identity || print(io, ", project=", l.proj)
    print(io, ", aggr=", l.aggr)
    print(io, ", normalize=", l.normalize)
    print(io, ", #sample=", l.num_sample)
    print(io, ")")
end

"""
    MeanAggregator(in => out, σ=identity; normalize=true, project=false,
                   bias=true, num_sample=10, init=glorot_uniform)

SAGEConv with mean aggregator.

See also [`SAGEConv`](@ref).
"""
MeanAggregator(args...; kwargs...) = SAGEConv(args..., mean; kwargs...)

"""
    MeanAggregator(in => out, σ=identity; normalize=true,
                   bias=true, num_sample=10, init=glorot_uniform)

SAGEConv with meanpool aggregator.

See also [`SAGEConv`](@ref).
"""
MeanPoolAggregator(args...; kwargs...) = SAGEConv(args..., mean; project=true, kwargs...)

"""
    MeanAggregator(in => out, σ=identity; normalize=true,
                   bias=true, num_sample=10, init=glorot_uniform)

SAGEConv with maxpool aggregator.

See also [`SAGEConv`](@ref).
"""
MaxPoolAggregator(args...; kwargs...) = SAGEConv(args..., max; project=true, kwargs...)


"""
    LSTMAggregator(in => out, σ=identity; normalize=true, project=false,
                   bias=true, num_sample=10, init=glorot_uniform)

SAGEConv with LSTM aggregator.

See also [`SAGEConv`](@ref).
"""
function LSTMAggregator(args...; kwargs...)
    in_ch = args[1][1]
    return SAGEConv(args..., Flux.LSTMCell(in_ch, in_ch); kwargs...)
end
