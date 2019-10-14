abstract type ContinuousMessagePassing <: MessagePassing end
abstract type ODEMessagePassing <: ContinuousMessagePassing end

struct ODEGraphConv{A,T,S<:AbstractVector} <: ODEMessagePassing
    adjlist::A
    V::AbstractVector{T}
    k::AbstractVector{T}
    weight1::AbstractVector{S}
    weight2::AbstractVector{S}
    bias1::AbstractVector{S}
    bias2::AbstractVector{S}
    aggr::Symbol
end

function ODEGraphConv(adj::AbstractMatrix, aggr::Symbol=:prod)
    ODEGraphConv(neighbors(adj), aggr)
end

function message(o::ODEGraphConv; x_i=zeros(0), x_j=zeros(0))
    h_1 = σ(o.weight1[i] .* log(x_j) + o.bias1[i])
    h_2 = o.weight2[i] .* h_1 + o.bias2[i]
    h_2
end
update(o::ODEGraphConv; X=zeros(0), M=zeros(0)) = o.V[i] * M - o.k[i] * X
function (ogc::ODEGraphConv)(X::AbstractMatrix, args...; kwargs...)
    nn = propagate(ogc, X=X, aggr=ogc.aggr)
    neural_ode(nn, X, tspan, args...; kwargs...)
end
