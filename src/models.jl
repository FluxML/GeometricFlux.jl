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



struct InnerProductDecoder
    σ
end

@functor InnerProductDecoder

(i::InnerProductDecoder)(Z::AbstractArray) = i.σ(Z'*Z)



struct VariationalEncoder
    nn
    μ
    logσ
    z_dim::Integer

    function VariationalEncoder(nn, h_dim::Integer, z_dim::Integer)
        new(nn, Dense(h_dim, z_dim), Dense(h_dim, z_dim), z_dim)
    end
end

@functor VariationalEncoder

function (ve::VariationalEncoder)(X::AbstractMatrix)
    h = ve.nn(X)
    μ, logσ = summarize(ve, h)
    Z = sampling(ve, μ, logσ)
    Z
end

summarize(ve::VariationalEncoder, h::AbstractMatrix) = (ve.μ(h), ve.logσ(h))
sampling(::VariationalEncoder, μ, logσ) = μ .+ exp.(logσ) .* randn(size(logσ))
