"""
    MessagePassing

An abstract type for message-passing scheme.

See also [`message`](@ref) and [`update`](@ref).
"""
abstract type MessagePassing <: GraphNet end

"""
    message(mp::MessagePassing, x_i, x_j, e_ij)

Message function for the message-passing scheme,
returning the message from node `j` to node `i` .
In the message-passing scheme. the incoming messages 
from the neighborhood of `i` will later be aggregated
in order to [`update`](@ref) the features of node `i`.

By default, the function returns `x_j`.
Layers subtyping [`MessagePassing`](@ref) should 
specialize this method with custom behavior.

# Arguments

- `mp`: message-passing layer.
- `x_i`: the features of node `i`.
- `x_j`: the features of the nighbor `j` of node `i`.
- `e_ij`: the features of edge (`i`, `j`).

See also [`update`](@ref).
"""
function message end

@inline message(mp::MessagePassing, x_i, x_j, e_ij) = x_j
@inline message(mp::MessagePassing, i::Integer, j::Integer, x_i, x_j, e_ij) = x_j

"""
    update(mp::MessagePassing, m, x)

Update function for the message-passing scheme,
returning a new set of node features `x′` based on old 
features `x` and the incoming message from the neighborhood
aggregation `m`.

By default, the function returns `m`.
Layers subtyping [`MessagePassing`](@ref) should 
specialize this method with custom behavior.

# Arguments

- `mp`: message-passing layer.
- `m`: the aggregated edge messages from the [`message`](@ref) function.
- `x`: the node features to be updated.

See also [`message`](@ref).
"""
function update end

@inline update(mp::MessagePassing, m, x) = m
@inline update(mp::MessagePassing, i::Integer, m, x) = m

update_edge(mp::MessagePassing, e, vi, vj, u) = GeometricFlux.message(mp, vi, vj, e)
update_vertex(mp::MessagePassing, ē, vi, u) = GeometricFlux.update(mp, ē, vi)

# For static graph
function WithGraph(fg::AbstractFeaturedGraph, mp::MessagePassing)
    return WithGraph(to_namedtuple(fg), mp)
end

(wg::WithGraph{<:MessagePassing})(X::AbstractArray) = wg.layer(wg.graph, X)
