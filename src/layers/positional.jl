"""
    AbstractPE

Abstract type of positional encoding for GNN.
"""
abstract type AbstractPE end

positional_encode(l::AbstractPE, args...) = throw(ErrorException("positional_encode function for $l is not implemented."))

struct EEquivGraphPE{X} <: MessagePassing
    nn_x::X
end

function EEquivGraphPE(ch::Pair{Int,Int}; init=glorot_uniform, bias::Bool=true)
    in, out = ch
    nn_x = Flux.Dense(in, out; init=init, bias=bias)
    return EEquivGraphPE(nn_x)
end

@functor EEquivGraphPE

ϕ_x(l::EEquivGraphPE, m_ij) = l.nn_x(m_ij)

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
function(l::EEquivGraphPE)(el::NamedTuple, X::AbstractArray, E::AbstractArray)
    GraphSignals.check_num_nodes(el.N, X)
    GraphSignals.check_num_nodes(el.E, E)
    _, V, _ = propagate(l, el, E, X, nothing, mean, nothing, nothing)
    return V
end

function Base.show(io::IO, l::EEquivGraphPE)
    print(io, "EEquivGraphPE(", input_dim(l), " => ", output_dim(l), ")")
end

input_dim(l::EEquivGraphPE) = size(l.nn_x.weight, 2)
output_dim(l::EEquivGraphPE) = size(l.nn_x.weight, 1)
