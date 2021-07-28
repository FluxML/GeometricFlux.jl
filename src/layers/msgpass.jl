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

"""
update(mp::GraphNet, m, x)

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
function message end

function propagate(mp::MessagePassing, fg::FeaturedGraph, aggr=+)
    E, X = propagate(mp, edge_index(fg), edge_feature(fg), node_feature(fg), aggr)
    FeaturedGraph(fg, nf=X, ef=E)
end

function propagate(mp::MessagePassing, eindex::Tuple, E, X, aggr)
    E, X, u = propagate(mp, eindex, E, X, nothing, aggr, nothing, nothing)
    E, X
end
