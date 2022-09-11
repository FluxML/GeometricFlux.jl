"""
    AbstractPositionalEncoding

Abstract type of positional encoding for GNN.
"""
abstract type AbstractPositionalEncoding end

"""
    RandomWalkPE{K}

Concrete type of positional encoding from random walk method.

See also [`positional_encode`](@ref) for generating positional encoding.
"""
struct RandomWalkPE{K} <: AbstractPositionalEncoding end

"""
    LaplacianPE{K}

Concrete type of positional encoding from graph Laplacian method.

See also [`positional_encode`](@ref) for generating positional encoding.
"""
struct LaplacianPE{K} <: AbstractPositionalEncoding end

"""
    positional_encode(RandomWalkPE{K}, A)

Returns positional encoding (PE) of size `(K, N)` where N is node number.
PE is generated by `K`-step random walk over given graph.

# Arguments

- `K::Int`: First dimension of PE.
- `A`: Adjacency matrix of a graph.

See also [`RandomWalkPE`](@ref) for random walk method.
"""
function positional_encode(::Type{RandomWalkPE{K}}, A::AbstractMatrix) where {K}
    N = size(A, 1)
    @assert K ≤ N "K=$K must less or equal to number of nodes ($N)"
    inv_D = GraphSignals.degree_matrix(A, Float32, inverse=true)

    RW = similar(A, size(A)..., K)
    RW[:, :, 1] .= A * inv_D
    for i in 2:K
        RW[:, :, i] .= RW[:, :, i-1] * RW[:, :, 1]
    end

    pe = similar(RW, K, N)
    for i in 1:N
        pe[:, i] .= RW[i, i, :]
    end

    return pe
end

"""
    positional_encode(LaplacianPE{K}, A)

Returns positional encoding (PE) of size `(K, N)` where `N` is node number.
PE is generated from eigenvectors of a graph Laplacian truncated by `K`.

# Arguments

- `K::Int`: First dimension of PE.
- `A`: Adjacency matrix of a graph.

See also [`LaplacianPE`](@ref) for graph Laplacian method.
"""
function positional_encode(::Type{LaplacianPE{K}}, A::AbstractMatrix) where {K}
    N = size(A, 1)
    @assert K ≤ N "K=$K must less or equal to number of nodes ($N)"
    L = GraphSignals.normalized_laplacian(A)
    U = eigvecs(L)
    return U[1:K, :]
end


"""
    EEquivGraphPE(in_dim=>out_dim; init=glorot_uniform, bias=true)

E(n)-equivariant positional encoding layer.

# Arguments

- `in_dim::Int`: dimension of input positional feature.
- `out_dim::Int`:  dimension of output positional feature.
- `init`: neural network initialization function.
- `bias::Bool`: dimension of edge feature.

# Examples

```jldoctest
julia> in_dim_edge, out_dim = 2, 5
(2, 5)

julia> l = EEquivGraphPE(in_dim_edge=>out_dim)
EEquivGraphPE(2 => 5)
```

See also [`EEquivGraphConv`](@ref).
"""
struct EEquivGraphPE{X} <: MessagePassing
    nn::X
end

function EEquivGraphPE(ch::Pair{Int,Int}; init=glorot_uniform, bias::Bool=true)
    in, out = ch
    nn = Flux.Dense(in, out; init=init, bias=bias)
    return EEquivGraphPE(nn)
end

@functor EEquivGraphPE

ϕ_x(l::EEquivGraphPE, m_ij) = l.nn(m_ij)

message(l::EEquivGraphPE, x_i, x_j, e) = (x_i - x_j) .* ϕ_x(l, e)

update(l::EEquivGraphPE, m::AbstractArray, x::AbstractArray) = m .+ x

# For variable graph
function(l::EEquivGraphPE)(fg::AbstractFeaturedGraph)
    X = node_feature(fg)
    E = edge_feature(fg)
    GraphSignals.check_num_nodes(fg, X)
    GraphSignals.check_num_nodes(fg, E)
    _, V, _ = propagate(l, graph(fg), E, X, nothing, mean, nothing, nothing)
    return ConcreteFeaturedGraph(fg, nf=V)
end

# For static graph
function(l::EEquivGraphPE)(el::NamedTuple, H::AbstractArray, E::AbstractArray)
    GraphSignals.check_num_nodes(el.N, H)
    # GraphSignals.check_num_edges(el.E, E)
    _, H, _ = propagate(l, el, E, H, nothing, mean, nothing, nothing)
    return H
end

(wg::WithGraph{<:EEquivGraphPE})(args...) = wg.layer(wg.graph, args...)

function Base.show(io::IO, l::EEquivGraphPE)
    print(io, "EEquivGraphPE(", input_dim(l), " => ", output_dim(l), ")")
end

input_dim(l::EEquivGraphPE) = size(l.nn.weight, 2)
output_dim(l::EEquivGraphPE) = size(l.nn.weight, 1)

positional_encode(wg::WithGraph{<:EEquivGraphPE}, args...) = wg(args...)
positional_encode(l::EEquivGraphPE, args...) = l(args...)


## LSPE

"""
    LSPE(graph, k; init_method=RandomWalkPE)

Learnable structural positional encoding layer which adds learnable positional encoding to
input data.

# Arguments

- `graph`: A given graph for positional encoding.
- `k::Int`: Dimension of positional encoding.
- `init_method`: Initializer for positional encoding.

See also [`GatedGCNLSPEConv`](@ref) layer and [`laplacian_eig_loss`](@ref) for positional loss.
"""
struct LSPE{P} <: AbstractPositionalEncoding
    pe::P
end

@functor LSPE

function LSPE(graph, k::Int; init_method=RandomWalkPE)
    fg = FeaturedGraph(graph)
    A = GraphSignals.adjacency_matrix(fg)
    pe = positional_encode(init_method{k}, A)
    return LSPE(pe)
end

positional_encode(l::LSPE) = l.pe

# For variable graph
function (l::LSPE)(fg::AbstractFeaturedGraph)
    if GraphSignals.has_positional_feature(fg)
        return fg
    else
        return ConcreteFeaturedGraph(fg, pf=positional_encode(l))
    end
end

Base.show(io::IO, l::LSPE) = print(io, "LSPE($(size(l.pe)))")


"""
    GatedGCNLSPEConv(in_dim => out_dim, pos_dim, σ=relu; residual=false, init=glorot_uniform)

Gated graph convolutional network layer with LSPE.

# Arguments

- `in_dim::Int`: The dimension of input features.
- `out_dim::Int`: The dimension of output features.
- `pos_dim::Int`: The dimension of positional encoding.
- `σ`: Activation function.
- `residual::Bool`: Add a skip connection introduced in ResNet.
- `init`: Weights' initialization function.

# Examples

```jldoctest
julia> GatedGCNLSPEConv(3 => 5, 4)
GatedGCNLSPEConv(3 => 5, positional dim=4, relu, residual=false)

julia> GatedGCNLSPEConv(3 => 5, 4, residual=true)
GatedGCNLSPEConv(3 => 5, positional dim=4, relu, residual=true)
```

See also [`LSPE`](@ref) for learnable positional encodings.
"""
struct GatedGCNLSPEConv{H,I,J,K,L,M,N,F} <: AbstractGraphLayer
    A1::H
    A2::I
    B1::J
    B2::K
    B3::L
    C1::M
    C2::N
    σ::F
    residual::Bool
end

@functor GatedGCNLSPEConv

function GatedGCNLSPEConv(ch::Pair{Int,Int}, pos_dim::Int, σ=relu; residual::Bool=false, init=glorot_uniform)
    in_dim, out_dim = ch
    A1 = Dense(in_dim + pos_dim, out_dim; init=init)
    A2 = Dense(in_dim + pos_dim, out_dim; init=init)
    B1 = Dense(in_dim, out_dim; init=init)
    B2 = Dense(in_dim, out_dim; init=init)
    B3 = Dense(in_dim, out_dim; init=init)
    C1 = Dense(in_dim, out_dim; init=init)
    C2 = Dense(in_dim, out_dim; init=init)
    return GatedGCNLSPEConv(A1, A2, B1, B2, B3, C1, C2, σ, residual)
end

# For variable graph
function (l::GatedGCNLSPEConv)(fg::AbstractFeaturedGraph)
    H = node_feature(fg)
    E = edge_feature(fg)
    X = positional_feature(fg)
    GraphSignals.has_node_feature(fg) && GraphSignals.check_num_nodes(fg, H)
    GraphSignals.has_positional_feature(fg) && GraphSignals.check_num_nodes(fg, X)
    GraphSignals.has_edge_feature(fg) && GraphSignals.check_num_edges(fg, E)
    E, H, X = propagate(l, graph(fg), E, H, X)
    return ConcreteFeaturedGraph(fg, nf=H, ef=E, pf=X)
end

# For static graph
WithGraph(fg::AbstractFeaturedGraph, l::GatedGCNLSPEConv) =
    WithGraph(GraphSignals.to_namedtuple(fg), l, GraphSignals.NullDomain())

(wg::WithGraph{<:GatedGCNLSPEConv})(args...) = wg.layer(wg.graph, args...)

function (l::GatedGCNLSPEConv)(el::NamedTuple, H::AbstractArray, E::AbstractArray, X::AbstractArray)
    GraphSignals.check_num_nodes(el.N, H)
    GraphSignals.check_num_nodes(el.N, X)
    GraphSignals.check_num_edges(el.E, E)
    E, H, X = propagate(l, el, E, H, X)
    return H, E, X
end

update_edge(l::GatedGCNLSPEConv, h_i, h_j, e_ij) = σ.(l.B1(h_i) + l.B2(h_j) + l.B3(e_ij))

function normalize_η(l::GatedGCNLSPEConv, el::NamedTuple, η̂)
    summed_η = aggregate_neighbors(l, el, +, η̂)
    return η̂ ./ gather(summed_η .+ 1f-6, el.xs)
end

message_vertex(l::GatedGCNLSPEConv, h_j, p_j, η_ij) = η_ij .* l.A2(vcat(h_j, p_j))

aggregate_neighbors(l::GatedGCNLSPEConv, el::NamedTuple, aggr, E) = scatter(aggr, E, el.xs, el.N)
aggregate_neighbors(l::GatedGCNLSPEConv, el::NamedTuple, aggr, E::AbstractMatrix) = scatter(aggr, E, el.xs)

update_vertex(l::GatedGCNLSPEConv, m, h, p) = l.σ.(l.A1(vcat(h, p)) + m)

message_position(l::GatedGCNLSPEConv, p_j, η_ij) = η_ij .* l.C2(p_j)

update_position(l::GatedGCNLSPEConv, m, p) = NNlib.tanh_fast.(l.C1(p) + m)

propagate(l::GatedGCNLSPEConv, sg::SparseGraph, E, H, X) =
    propagate(l, GraphSignals.to_namedtuple(sg), E, H, X)

function propagate(l::GatedGCNLSPEConv, el::NamedTuple, E, H, X)
    e_ij = gather(E, el.es)
    h_i = gather(H, el.xs)
    h_j = gather(H, el.nbrs)
    p_i = gather(X, el.xs)
    p_j = gather(X, el.nbrs)

    η̂ = update_edge(l, h_i, h_j, e_ij)
    Ê = l.σ.(η̂)
    η_ij = normalize_η(l, el, η̂)

    m_h = message_vertex(l, h_j, p_j, η_ij)
    m_h = aggregate_neighbors(l, el, +, m_h)
    Ĥ = update_vertex(l, m_h, H, X)

    m_p = message_position(l, p_j, η_ij)
    m_p = aggregate_neighbors(l, el, +, m_p)
    X̂ = update_position(l, m_p, X)

    if l.residual
        Ĥ = H + Ĥ
        X̂ = X + X̂
    end
    return E, H, X
end

function Base.show(io::IO, l::GatedGCNLSPEConv)
    out_dim, in_dim = size(l.B1.weight)
    pos_dim = size(l.A1.weight, 2) - in_dim
    print(io, "GatedGCNLSPEConv(", in_dim, " => ", out_dim)
    print(io, ", positional dim=", pos_dim)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", residual=", l.residual)
    print(io, ")")
end
