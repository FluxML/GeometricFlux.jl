"""
    GCNConv([fg,] in => out, σ=identity; bias=true, init=glorot_uniform)

Graph convolutional layer.

# Arguments

- `fg`: Optionally pass a [`FeaturedGraph`](@ref). 
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

(l::GCNConv)(fg::FeaturedGraph) = FeaturedGraph(fg, nf = l(fg, node_feature(fg)))
(l::GCNConv)(x::AbstractMatrix) = l(l.fg, x)

function Base.show(io::IO, l::GCNConv)
    out, in = size(l.weight)
    print(io, "GCNConv($in => $out")
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end


"""
    ChebConv([fg,] in=>out, k; bias=true, init=glorot_uniform)

Chebyshev spectral graph convolutional layer.

# Arguments

- `fg`: Optionally pass a [`FeaturedGraph`](@ref). 
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
end

function ChebConv(fg::AbstractFeaturedGraph, ch::Pair{Int,Int}, k::Int;
                  init=glorot_uniform, bias::Bool=true)
    in, out = ch
    W = init(out, in, k)
    b = Flux.create_bias(W, bias, out)
    ChebConv(W, b, fg, k)
end

ChebConv(ch::Pair{Int,Int}, k::Int; kwargs...) =
    ChebConv(NullGraph(), ch, k; kwargs...)

@functor ChebConv

function (c::ChebConv)(fg::FeaturedGraph, X::AbstractMatrix{T}) where T
    check_num_nodes(fg, X)
    @assert size(X, 1) == size(c.weight, 2) "Input feature size must match input channel size."
    
    L̃ = scaled_laplacian(fg, eltype(X))    

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

(l::ChebConv)(fg::FeaturedGraph) = FeaturedGraph(fg, nf = l(fg, node_feature(fg)))
(l::ChebConv)(x::AbstractMatrix) = l(l.fg, x)

function Base.show(io::IO, l::ChebConv)
    out, in, k = size(l.weight)
    print(io, "ChebConv(", in, " => ", out)
    print(io, ", k=", k)
    print(io, ")")
end


"""
    GraphConv([fg,] in => out, σ=identity, aggr=+; bias=true, init=glorot_uniform)

Graph neural network layer.

# Arguments

- `fg`: Optionally pass a [`FeaturedGraph`](@ref). 
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

message(gc::GraphConv, x_i, x_j, e_ij) =  x_j

update(gc::GraphConv, m, x) = gc.σ.(gc.weight1 * x .+ gc.weight2 * m .+ gc.bias)

function (gc::GraphConv)(fg::FeaturedGraph, x::AbstractMatrix)
    check_num_nodes(fg, x)
    _, x = propagate(gc, fg, nothing, x, +)
    x
end

(l::GraphConv)(fg::FeaturedGraph) = FeaturedGraph(fg, nf = l(fg, node_feature(fg)))
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
    GATConv([fg,] in => out;
            heads=1,
            concat=true,
            init=glorot_uniform    
            bias=true, 
            negative_slope=0.2)

Graph attentional layer.

# Arguments

- `fg`: Optionally pass a [`FeaturedGraph`](@ref). 
- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `bias::Bool`: Keyword argument, whether to learn the additive bias.
- `heads`: Number attention heads 
- `concat`: Concatenate layer output or not. If not, layer output is averaged over the heads.
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

function (gat::GATConv)(fg::FeaturedGraph, X::AbstractMatrix)
    check_num_nodes(fg, X)
    # add_self_loop!(adj) #TODO
    chin, chout = gat.channel
    heads = gat.heads

    source, target = edge_index(fg)
    Wx = gat.weight*X
    Wx = reshape(Wx, chout, heads, :)                   # chout × nheads × nnodes
    Wxi = NNlib.gather(Wx, target)                      # chout × nheads × nedges
    Wxj = NNlib.gather(Wx, source)

    # Edge Message
    # Computing softmax. TODO make it numerically stable
    aWW = sum(gat.a .* cat(Wxi, Wxj, dims=1), dims=1)   # 1 × nheads × nedges
    α = exp.(leakyrelu.(aWW, gat.negative_slope))       
    m̄ =  NNlib.scatter(+, α .* Wxj, target)             # chout × nheads × nnodes 
    ᾱ = NNlib.scatter(+, α, target)                     # 1 × nheads × nnodes
    
    # Node update
    b = reshape(gat.bias, chout, heads)
    X = m̄ ./ ᾱ .+ b                                     # chout × nheads × nnodes 
    if !gat.concat
        X = sum(X, dims=2)
    end

    # We finally return a matrix
    return reshape(X, :, size(X, 3)) 
end

(l::GATConv)(fg::FeaturedGraph) = FeaturedGraph(fg, nf = l(fg, node_feature(fg)))
(l::GATConv)(x::AbstractMatrix) = l(l.fg, x)

function Base.show(io::IO, l::GATConv)
    in_channel = size(l.weight, ndims(l.weight))
    out_channel = size(l.weight, ndims(l.weight)-1)
    print(io, "GATConv(", in_channel, "=>", out_channel)
    print(io, ", LeakyReLU(λ=", l.negative_slope)
    print(io, "))")
end


"""
    GatedGraphConv([fg,] out, num_layers; aggr=+, init=glorot_uniform)

Gated graph convolution layer.

# Arguments

- `fg`: Optionally pass a [`FeaturedGraph`](@ref). 
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

message(ggc::GatedGraphConv, x_i, x_j, e_ij) = x_j

update(ggc::GatedGraphConv, m, x) = m


function (ggc::GatedGraphConv)(fg::FeaturedGraph, H::AbstractMatrix{S}) where {T<:AbstractVector,S<:Real}
    check_num_nodes(fg, H)
    m, n = size(H)
    @assert (m <= ggc.out_ch) "number of input features must less or equals to output features."
    if m < ggc.out_ch
        Hpad = similar(H, S, ggc.out_ch - m, n)
        H = vcat(H, fill!(Hpad, 0))
    end
    for i = 1:ggc.num_layers
        M = view(ggc.weight, :, :, i) * H
        _, M = propagate(ggc, fg, nothing, M, +)
        H, _ = ggc.gru(H, M)
    end
    H
end

(l::GatedGraphConv)(fg::FeaturedGraph) = FeaturedGraph(fg, nf = l(fg, node_feature(fg)))
(l::GatedGraphConv)(x::AbstractMatrix) = l(l.fg, x)


function Base.show(io::IO, l::GatedGraphConv)
    print(io, "GatedGraphConv(($(l.out_ch) => $(l.out_ch))^$(l.num_layers)")
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end



"""
    EdgeConv([fg,] nn; aggr=max)

Edge convolutional layer.

# Arguments

- `fg`: Optionally pass a [`FeaturedGraph`](@ref). 
- `nn`: A neural network (e.g. a Dense layer or a MLP). 
- `aggr`: An aggregate function applied to the result of message function. `+`, `max` and `mean` are available.
"""
struct EdgeConv{V<:AbstractFeaturedGraph} <: MessagePassing
    fg::V
    nn
    aggr
end

EdgeConv(fg::AbstractFeaturedGraph, nn; aggr=max) = EdgeConv(fg, nn, aggr)
EdgeConv(nn; kwargs...) = EdgeConv(NullGraph(), nn; kwargs...)

@functor EdgeConv

message(ec::EdgeConv, x_i, x_j, e_ij) = ec.nn(vcat(x_i, x_j .- x_i))

update(ec::EdgeConv, m, x) = m

function (ec::EdgeConv)(fg::FeaturedGraph, X::AbstractMatrix)
    check_num_nodes(fg, X)
    _, X = propagate(ec, fg, nothing, X, ec.aggr)
    X
end

(l::EdgeConv)(fg::FeaturedGraph) = FeaturedGraph(fg, nf = l(fg, node_feature(fg)))
(l::EdgeConv)(x::AbstractMatrix) = l(l.fg, x)

function Base.show(io::IO, l::EdgeConv)
    print(io, "EdgeConv(", l.nn)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end
