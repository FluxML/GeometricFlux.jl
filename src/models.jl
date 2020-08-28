"""
    GAE(enc[, σ])

Graph autoencoder.

# Arguments
- `enc`: encoder. It can be any graph convolutional layer.

Encoder is specified by user and decoder will be `InnerProductDecoder` layer.
"""
struct GAE{T,S}
    encoder::T
    decoder::S
end

GAE(enc, σ::Function=identity) = GAE(enc, InnerProductDecoder(σ))

@functor GAE

function (g::GAE)(X::AbstractMatrix)
    Z = g.encoder(X)
    A = g.decoder(Z)
    A
end


"""
    VGAE(enc[, σ])

Variational graph autoencoder.

# Arguments
- `enc`: encoder. It can be any graph convolutional layer.

Encoder is specified by user and decoder will be `InnerProductDecoder` layer.
"""
struct VGAE{T,S}
    encoder::T
    decoder::S
end

function VGAE(enc, h_dim::Integer, z_dim::Integer, σ::Function=identity)
    VGAE(VariationalEncoder(enc, h_dim, z_dim), InnerProductDecoder(σ))
end

@functor VGAE

function (g::VGAE)(X::AbstractMatrix)
    Z = g.encoder(X)
    A = g.decoder(Z)
    A
end

function (g::VGAE)(fg::FeaturedGraph)
    Z = g.encoder(X)
    A = g.decoder(Z)
    A
end


"""
    InnerProductDecoder(σ)

Inner-product decoder layer.

# Arguments
- `σ`: activation function.
"""
struct InnerProductDecoder
    σ
end

@functor InnerProductDecoder

(i::InnerProductDecoder)(Z::AbstractMatrix)::AbstractMatrix = i.σ.(Z'*Z)

function (i::InnerProductDecoder)(fg::FeaturedGraph)::FeaturedGraph
    Z = node_feature(fg)
    A = i(Z)
    FeaturedGraph(graph(fg), A)
end


"""
    VariationalEncoder(nn, h_dim, z_dim)

Variational encoder layer.

# Arguments
- `nn`: neural network. It can be any graph convolutional layer.
- `h_dim`: dimension of hidden layer. This should fit the output dimension of `nn`.
- `z_dim`: dimension of latent variable layer. This will be parametrized into `μ` and `logσ`.

Encoder can be any graph convolutional layer.
"""
struct VariationalEncoder
    nn
    μ
    logσ
    z_dim::Integer
end

function VariationalEncoder(nn, h_dim::Integer, z_dim::Integer)
    VariationalEncoder(nn, Dense(h_dim, z_dim), Dense(h_dim, z_dim), z_dim)
end

@functor VariationalEncoder

function (ve::VariationalEncoder)(X::AbstractMatrix)::AbstractMatrix
    μ, logσ = summarize(ve, X)
    Z = sample(μ, logσ)
    Z
end

function (ve::VariationalEncoder)(fg::FeaturedGraph)::FeaturedGraph
    μ, logσ = summarize(ve, fg)
    Z = sample(μ, logσ)
    FeaturedGraph(graph(fg), Z)
end

function summarize(ve::VariationalEncoder, X::AbstractMatrix)
    h = ve.nn(X)
    ve.μ(h), ve.logσ(h)
end

function summarize(ve::VariationalEncoder, fg::FeaturedGraph)
    fg_ = ve.nn(fg)
    h = node_feature(fg_)
    ve.μ(h), ve.logσ(h)
end

sample(μ::AbstractArray{T}, logσ::AbstractArray{T}) where {T<:Real} =
    μ + exp.(logσ) .* randn(T, size(logσ))
