# struct MessagePassing{L1,L2,L3}
#     msg_func::L1
#     upd_func::L2
#     aggr::L3
# end
#
# MessagePassing(msg_func, upd_func, aggr=+) = MessagePassing(msg_func, upd_func, aggr)
#
# (mp::MessagePassing)(xᵢ, xⱼ, eᵢⱼ) = update(mp, xᵢ, m.aggr(message(mp, xᵢ, xⱼ, eᵢⱼ)...))
#
# message(m::MessagePassing, xᵢ, xⱼ, eᵢⱼ) = m.msg_func(xᵢ, xⱼ, eᵢⱼ)
# update(m::MessagePassing, xᵢ, mᵢ) = m.upd_func(xᵢ, mᵢ)
#
# function Base.show(io::IO, m::MessagePassing)
#     print(io, "MessagePassing(msg:", m.msg_func)
#     print(io, ", upd:", m.upd_func)
#     print(io, ", aggr:", m.aggr, ")")
# end

struct GCNConv{A,F}
    weight::A
    norm::A
    σ::F
    GCNConv(w::AbstractMatrix{T}, n::AbstractMatrix{T}, σ = identity) where T = new(w, n, σ)
end

function GCNConv(g, ch::Pair{<:Integer,<:Integer}, σ = identity; init = glorot_uniform)
    GCNConv(param(init(k..., ch...)), param(zeros(ch[2])), σ)
end

(c::GCNConv)(X::AbstractMatrix) = c.σ(c.norm * X * c.weight)

# GraphConv(ch::Pair{<:Integer,<:Integer}, aggr::Symbol, bias::Bool)
#
# ChebConv(ch::Pair{<:Integer,<:Integer}, k::Integer)
#
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
