struct MessagePassing{L1,L2,L3}
    msg_func::L1
    upd_func::L2
    aggr::L3
end

MessagePassing(msg_func, upd_func, aggr=+) = MessagePassing(msg_func, upd_func, aggr)

(mp::MessagePassing)(xᵢ, xⱼ, eᵢⱼ) = update(mp, xᵢ, m.aggr(message(mp, xᵢ, xⱼ, eᵢⱼ)...))

message(m::MessagePassing, xᵢ, xⱼ, eᵢⱼ) = m.msg_func(xᵢ, xⱼ, eᵢⱼ)
update(m::MessagePassing, xᵢ, mᵢ) = m.upd_func(xᵢ, mᵢ)

function Base.show(io::IO, m::MessagePassing)
    print(io, "MessagePassing(msg:", m.msg_func)
    print(io, ", upd:", m.upd_func)
    print(io, ", aggr:", m.aggr, ")")
end



struct GCNConv{A,F}
    weight::A
    norm::A
    σ::F
    GCNConv(w::AbstractMatrix{T}, n::AbstractMatrix{T}, σ = identity) where T = new(w, n, σ)
end

function GCNConv(adj::AbstractMatrix{T}, ch::Pair{<:Integer,<:Integer}, σ = identity; init = glorot_uniform)
    # TODO
end

function GCNConv(g::AbstractSimpleGraph, ch::Pair{<:Integer,<:Integer}, σ = identity; init = glorot_uniform)
    GCNConv(param(init(k..., ch...)), param(zeros(ch[2])), σ)
end

(c::GCNConv)(X::AbstractMatrix) = c.σ(c.norm * X * c.weight)



struct ChebConv{A,}
    weight::A
    Zs::
    L::
    k::Integer
    ChebConv(w::AbstractMatrix{T}, Zs::AbstractArray{T,3}, L::AbstractMatrix{T},
             k::Int) where T = new(w, Zs, L, k)
end

function ChebConv(adj::AbstractMatrix{T}, ch::Pair{<:Integer,<:Integer}, k::Integer,
    σ = identity; init = glorot_uniform)
    # TODO
end

function ChebConv(g::AbstractSimpleGraph, ch::Pair{<:Integer,<:Integer}, k::Integer,
    σ = identity; init = glorot_uniform)
    return ChebConv(, zeros(size(X, 1), size(X, 2), k))
end

function (c::ChebConv)(X::AbstractMatrix)
    c.Zs[:,:,1] = c.L * X
    c.Zs[:,:,2] =  2*c.L*c.Zs[:,:,1] - X
    for i = 3:c.k
        c.Zs[:,:,i] = 2*c.L*c.Zs[:,:,i-1] - c.Zs[:,:,i-2])
    end

    X_ = copy(X)
    for i = 1:(c.k-1)
        X_ += c.Zs[:,:,i]*c.weight[i]
    end
    X_
end

# GraphConv(ch::Pair{<:Integer,<:Integer}, aggr::Symbol, bias::Bool)

# GatedGraphConv(out_ch::Integer, len::Integer, aggr::Symbol, bias::Bool)
#
# GATConv(in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True)
#
# EdgeConv(nn, aggr::Symbol)
#
# Meta(edge_model, node_model, global_model)

# Pooling Layers
# GlobalPool(x, aggr, batch, size=None) # aggr=sum, mean, max
#
# TopKPool()
#
# MaxPool()
#
# MeanPool()

# InnerProductDecoder()
