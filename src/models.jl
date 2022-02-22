"""
    GAE(enc, [σ=identity])

Graph autoencoder.

# Arguments
- `enc`: encoder. It can be any graph convolutional layer.
- `σ`: Activation function for decoder.

Encoder is specified by user and decoder will be `InnerProductDecoder` layer.
"""
struct GAE{T,S} <: AbstractGraphLayer
    encoder::T
    decoder::S
end

GAE(enc, σ::Function=identity) = GAE(enc, InnerProductDecoder(σ))

@functor GAE

# For variable graph
(l::GAE)(fg::AbstractFeaturedGraph) = fg |> l.encoder |> l.decoder

# For static graph
WithGraph(fg::AbstractFeaturedGraph, l::GAE) = GAE(WithGraph(fg, l.encoder), l.decoder)

(l::GAE)(X::AbstractMatrix) = X |> l.encoder |> l.decoder
(l::GAE)(X::AbstractArray) = X |> l.encoder |> l.decoder


"""
    VGAE(enc[, σ])

Variational graph autoencoder.

# Arguments
- `enc`: encoder. It can be any graph convolutional layer.

Encoder is specified by user and decoder will be `InnerProductDecoder` layer.
"""
struct VGAE{T,S} <: AbstractGraphLayer
    encoder::T
    decoder::S
end

function VGAE(enc, h_dim::Integer, z_dim::Integer, σ::Function=identity)
    VGAE(VariationalGraphEncoder(enc, h_dim, z_dim), InnerProductDecoder(σ))
end

@functor VGAE

# For variable graph
(l::VGAE)(fg::AbstractFeaturedGraph) = fg |> l.encoder |> l.decoder

# For static graph
WithGraph(fg::AbstractFeaturedGraph, l::VGAE) = VGAE(WithGraph(fg, l.encoder), l.decoder)

(l::VGAE)(X::AbstractArray) = X |> l.encoder |> l.decoder


"""
    InnerProductDecoder(σ)

Inner-product decoder layer.

# Arguments
- `σ`: activation function.
"""
struct InnerProductDecoder{F}
    σ::F
end

@functor InnerProductDecoder

(i::InnerProductDecoder)(Z::AbstractMatrix)::AbstractMatrix = i.σ.(Z'*Z)
(i::InnerProductDecoder)(Z::AbstractArray)::AbstractArray =
    i.σ.(NNlib.batched_mul(NNlib.batched_transpose(Z), Z))

function (i::InnerProductDecoder)(fg::FeaturedGraph)
    Z = node_feature(fg)
    A = i(Z)
    return FeaturedGraph(fg, nf=A)
end


"""
    VariationalGraphEncoder(nn, h_dim, z_dim)

Variational graph encoder layer.

# Arguments
- `nn`: neural network. It can be any graph convolutional layer.
- `h_dim`: dimension of hidden layer. This should fit the output dimension of `nn`.
- `z_dim`: dimension of latent variable layer. This will be parametrized into `μ` and `logσ`.

Encoder can be any graph convolutional layer.
"""
struct VariationalGraphEncoder{L,M,S,T<:Integer} <: AbstractGraphLayer
    nn::L
    μ::M
    logσ::S
    z_dim::T
end

function VariationalGraphEncoder(nn, h_dim::Integer, z_dim::Integer)
    VariationalGraphEncoder(nn,
                       GCNConv(h_dim=>z_dim),
                       GCNConv(h_dim=>z_dim),
                       z_dim)
end

@functor VariationalGraphEncoder

function (ve::VariationalGraphEncoder)(fg::FeaturedGraph)::FeaturedGraph
    μ, logσ = summarize(ve, fg)
    Z = sample(μ, logσ)
    FeaturedGraph(fg, nf=Z)
end

function summarize(ve::VariationalGraphEncoder, fg::FeaturedGraph)
    fg_ = ve.nn(fg)
    fg_μ, fg_logσ = ve.μ(fg_), ve.logσ(fg_)
    node_feature(fg_μ), node_feature(fg_logσ)
end

sample(μ::AbstractArray{T}, logσ::AbstractArray{T}) where {T<:Real} =
    μ + exp.(logσ) .* randn(T, size(logσ))

# For static graph
WithGraph(fg::AbstractFeaturedGraph, l::VariationalGraphEncoder) =
    VariationalGraphEncoder(
        WithGraph(fg, l.nn),
        WithGraph(fg, l.μ),
        WithGraph(fg, l.logσ),
        l.z_dim
    )

# (l::VariationalGraphEncoder)(X::AbstractArray) = X |> l.encoder |> l.decoder
