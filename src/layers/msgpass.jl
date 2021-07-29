# Adapted message passing from paper 
# "Relational inductive biases, deep learning, and graph networks"
abstract type MessagePassing end

function propagate(gn::MessagePassing, fg::FeaturedGraph, aggr=+)
    E, X, u = propagate(gn, fg,
                        edge_feature(fg), node_feature(fg), global_feature(fg), 
                        aggr)
    FeaturedGraph(fg, nf=X, ef=E, gf=u)
end

function propagate(gn::MessagePassing, fg::FeaturedGraph, E, X, u, aggr=+)
    M = compute_batch_message(gn, fg, E, X, u) 
    E = update_batch_edge(gn, M, E, u)
    M̄ = aggregate_neighbors(gn, aggr, fg, M)
    X = update_batch_vertex(gn, M̄, X, u)
    u = update_global(gn, E, X, u)
    return E, X, u
end

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


_gather(x, i) = NNlib.gather(x, i)
_gather(x::Nothing, i) = nothing

## Step 1.

function compute_batch_message(gn::MessagePassing, fg, E, X, u)
    s, t = edge_index(fg)
    Xi = _gather(X, t)
    Xj = _gather(X, s)
    M = message(gn, Xi, Xj, E, u)
    return M
end

# @inline message(gn::MessagePassing, i, j, x_i, x_j, e_ij, u) = message(gn, x_i, x_j, e_ij, u) # TODO add in the future
@inline message(gn::MessagePassing, x_i, x_j, e_ij, u) = message(gn, x_i, x_j, e_ij)
@inline message(gn::MessagePassing, x_i, x_j, e_ij) = message(gn, x_i, x_j)
@inline message(gn::MessagePassing, x_i, x_j) = x_j

## Step 2

function aggregate_neighbors(gn::MessagePassing, aggr, fg, E)
    s, t = edge_index(fg)
    NNlib.scatter(aggr, E, t)
end

aggregate_neighbors(gn::MessagePassing, aggr::Nothing, fg, E) = nothing

##  Step 3

update_batch_vertex(gn::MessagePassing, M̄, X, u) = update(gn, M̄, X, u)

# @inline update(gn::MessagePassing, i, m̄, x, u) = update(gn, m, x, u)
@inline update(gn::MessagePassing, m̄, x, u) = update(gn, m̄, x)
@inline update(gn::MessagePassing, m̄, x) = m̄

## Step 4
update_batch_edge(gn::MessagePassing, M, E, u) = update_edge(gn::MessagePassing, M, E, u)

@inline update_edge(gn::MessagePassing, M, E, u) = update_edge(gn::MessagePassing, M, E)
@inline update_edge(gn::MessagePassing, M, E) = E

## Step 5

@inline update_global(gn::MessagePassing, E, X, u) = u

### end steps ###

