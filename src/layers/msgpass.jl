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
@inline update(mp::MessagePassing, m, x) = m
@inline update(mp::MessagePassing, i::Integer, m, x) = m

@inline apply_batch_message(mp::MessagePassing, i, js, edge_idx, E::AbstractMatrix, X::AbstractMatrix, u) =
    mapreduce(j -> GeometricFlux.message(mp, _view(X, i), _view(X, j), _view(E, edge_idx[(i,j)])), hcat, js)

@inline update_batch_vertex(mp::MessagePassing, M::AbstractMatrix, X::AbstractMatrix, u) = 
    mapreduce(i -> GeometricFlux.update(mp, _view(M, i), _view(X, i)), hcat, 1:size(X,2))

function propagate(mp::MessagePassing, fg::FeaturedGraph, aggr=+)
    E, X = propagate(mp, adjacency_list(fg), fg.ef, fg.nf, aggr)
    FeaturedGraph(fg, nf=X, ef=E, gf=Fill(0.f0, 0))
end

function propagate(mp::MessagePassing, adj::AbstractVector{S}, E::R, X::Q, aggr) where {S<:AbstractVector,R,Q}
    E, X, u = propagate(mp, adj, E, X, Fill(0.f0, 0), aggr, nothing, nothing)
    E, X
end
