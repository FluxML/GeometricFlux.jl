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



# function logpdf(b::Bernoulli, y::Bool; T::Real=Float32)
#     y * log(b.p + eps(T)) + (one(T) - y) * log(one(T) - b.p + eps(T))
# end
#
# # KL-divergence between approximation posterior and N(0, 1) prior.
# kl_q_p(μ, logσ; T::Real=Float32) = T(0.5) * sum(exp.(T(2) .* logσ) + μ.^2 .- one(T) .+ logσ.^2)
#
# # logp(x|z) - decoder
# logp_x_z(x, z) = sum(logpdf.(Bernoulli.(f(z)), x))
#
# function loss(X; T=eltype(X), β=one(T), λ=T(0.01))
#     N = size(X, 1)  # batch size
#     μ̂, logσ̂ = g(X)
#     L̄ = 1//N * (logp_x_z(X, sampling.(μ̂, logσ̂)) - β * kl_q_p(μ̂, logσ̂))
#     -L̄ + λ * sum(x->sum(x.^2), params(f))
# end
