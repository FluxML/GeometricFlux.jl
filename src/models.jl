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

function (ve::VariationalGraphEncoder)(X::AbstractArray)::AbstractArray
    μ, logσ = summarize(ve, X)
    return sample(μ, logσ)
end

function summarize(ve::VariationalGraphEncoder, fg::FeaturedGraph)
    fg_ = ve.nn(fg)
    fg_μ, fg_logσ = ve.μ(fg_), ve.logσ(fg_)
    node_feature(fg_μ), node_feature(fg_logσ)
end

function summarize(ve::VariationalGraphEncoder, X::AbstractArray)
    H = ve.nn(X)
    return ve.μ(H), ve.logσ(H)
end

function sample(μ::AbstractArray{T}, logσ::AbstractArray{T}) where {T<:Real}
    R = Zygote.ignore(() -> randn!(similar(logσ)))
    return μ + exp.(logσ) .* R
end

# For static graph
WithGraph(fg::AbstractFeaturedGraph, l::VariationalGraphEncoder) =
    VariationalGraphEncoder(
        WithGraph(fg, l.nn),
        WithGraph(fg, l.μ),
        WithGraph(fg, l.logσ),
        l.z_dim
    )


"""
    DeepSet(ϕ, ρ, aggr=+)

Deep set model.

# Arguments

- `ϕ`: Neural network layer for each input before aggregation.
- `ρ`: Neural network layer after aggregation.
- `aggr`: An aggregate function applied to the result of message function. `+`, `-`,
`*`, `/`, `max`, `min` and `mean` are available.

# Examples

```jldoctest
julia> ϕ = Dense(64, 16)
Dense(64 => 16)     # 1_040 parameters

julia> ρ = Dense(16, 4)
Dense(16 => 4)      # 68 parameters

julia> DeepSet(ϕ, ρ)
DeepSet(Dense(64 => 16), Dense(16 => 4), aggr=+)

julia> DeepSet(ϕ, ρ, aggr=max)
DeepSet(Dense(64 => 16), Dense(16 => 4), aggr=max)
```

See also [`WithGraph`](@ref) for training layer with static graph.
"""
struct DeepSet{T,S,O} <: GraphNet
    ϕ::T
    ρ::S
    aggr::O
end

DeepSet(ϕ, ρ; aggr=+) = DeepSet(ϕ, ρ, aggr)

@functor DeepSet

update_batch_edge(l::DeepSet, el::NamedTuple, E, V, u) = nothing

update_vertex(l::DeepSet, Ē, V, u) = l.ϕ(V)

update_global(l::DeepSet, ē, v̄, u) = l.ρ(v̄)

# For variable graph
function (l::DeepSet)(fg::AbstractFeaturedGraph)
    X = node_feature(fg)
    u = global_feature(fg)
    GraphSignals.check_num_nodes(fg, X)
    _, _, u = propagate(l, graph(fg), nothing, X, u, nothing, nothing, l.aggr)
    return ConcreteFeaturedGraph(fg, gf=u)
end

# For static graph
function (l::DeepSet)(el::NamedTuple, X::AbstractArray, u=nothing)
    GraphSignals.check_num_nodes(el.N, X)
    _, _, u = propagate(l, el, nothing, X, u, nothing, nothing, l.aggr)
    return u
end

WithGraph(fg::AbstractFeaturedGraph, l::DeepSet) = WithGraph(to_namedtuple(fg), l)
(wg::WithGraph{<:DeepSet})(args...) = wg.layer(wg.graph, args...)

function Base.show(io::IO, l::DeepSet)
    print(io, "DeepSet(", l.ϕ, ", ", l.ρ)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end

function Base.show(io::IO, l::WithGraph{<:DeepSet})
    print(io, "WithGraph(Graph(#V=", l.graph.N)
    print(io, ", #E=", l.graph.E, "), ")
    print(io, l.layer, ")")
end
