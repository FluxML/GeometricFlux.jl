abstract type GraphNet end

_view(x::AbstractMatrix, i) = view(x, :, i)  # use standard indexing instead of views?
_view(x::Nothing, i) = nothing

aggregate(aggr::typeof(+), X::AbstractMatrix) = vec(sum(X, dims=2))
aggregate(aggr::typeof(-), X::AbstractMatrix) = -vec(sum(X, dims=2))
aggregate(aggr::typeof(*), X::AbstractMatrix) = vec(prod(X, dims=2))
aggregate(aggr::typeof(/), X::AbstractMatrix) = 1 ./ vec(prod(X, dims=2))
aggregate(aggr::typeof(max), X::AbstractMatrix) = vec(maximum(X, dims=2))
aggregate(aggr::typeof(min), X::AbstractMatrix) = vec(minimum(X, dims=2))
aggregate(aggr::typeof(mean), X::AbstractMatrix) = vec(mean(X, dims=2))
aggregate(aggr::Nothing, X::AbstractMatrix) = nothing

## Step 1.

function update_batch_edge(gn::GraphNet, st, E, X, u)
    s, t = st
    message(gn, X[:,t], X[:,s], E, u) # use view instead of indexing?
end

message(gn::GraphNet, x_i, x_j, e_ij, u) = x_j
# message(gn::GraphNet, i, j, x_i, x_j, e_ij, u) = message(gn, x_i, x_j, e_ij, u) # TODO add in the future

## Step 2

function aggregate_neighbors(gn::GraphNet, aggr, st, E)
    s, t = st
    NNlib.scatter(aggr, E, t)
end

aggregate_neighbors(gn::GraphNet, aggr::Nothing, st, E) = nothing

##  Step 3

update_batch_vertex(gn::GraphNet, M, X, u) = update(gn, M, X, u)
    
update(gn::GraphNet, m, x, u) = x
# update(gn::GraphNet, i, m, x, u) = update(gn, m, x, u)


## Step 4

aggregate_edges(gn::GraphNet, aggr, E) = aggregate(aggr, E)

## Step 5

aggregate_vertices(gn::GraphNet, aggr, X) = aggregate(aggr, X)

## Step 6

update_global(gn::GraphNet, ē, x̄, u) = u

### end steps ###


function propagate(gn::GraphNet, fg::FeaturedGraph, naggr=nothing, eaggr=nothing, vaggr=nothing)
    E, X, u = propagate(gn, edge_index(fg), 
                        edge_feature(fg), node_feature(fg), global_feature(fg), 
                        naggr, eaggr, vaggr)
    FeaturedGraph(fg, nf=X, ef=E, gf=u)
end

function propagate(gn::GraphNet, st::Tuple, E, X, u,
                   naggr=nothing, eaggr=nothing, vaggr=nothing)
    E = update_batch_edge(gn, st, E, X, u)
    @show E
    Ē = aggregate_neighbors(gn, naggr, st, E)
    @show Ē
    X = update_batch_vertex(gn, Ē, X, u)
    @show X
    ē = aggregate_edges(gn, eaggr, E)
    x̄ = aggregate_vertices(gn, vaggr, X)
    u = update_global(gn, ē, x̄, u)
    return E, X, u
end
